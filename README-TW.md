# vjepa-ac

[English](README.md)

這是一個小型的 **V-JEPA 2-AC** 學習與實驗專案。模型先預測機器人移動後，相機
畫面會如何改變，再利用預測結果規劃動作，讓畫面接近指定的目標圖片。

原版使用約 62 小時的機器人資料與 3 億參數的預測器。本專案只使用
`nvidia/Cosmos3-DROID` 前 100 段成功資料（約 45 分鐘），訓練約 2,400 萬個
參數，數小時的 GPU 時間即可完成。本機 CPU 可執行測試與快速檢查；完整訓練在
遠端 GPU 主機執行。

## 成果摘要

在這個資料規模下，直接預測 V-JEPA 的完整影像特徵沒有成功：模型幾乎不理會
機器人動作。最終設計先把每幀壓縮成 16 個能反映動作的 token，再於這個較小的
空間中預測。

| 未參與訓練的資料 | 原始特徵模型 | 最終模型 |
| --- | ---: | ---: |
| 打亂動作後的誤差增加 | +0.2% | 第 15 步 +50-65% |
| 模型誤差 / 重複輸入畫面的誤差 | 約 1.0 | 0.67-0.73 |
| 找回畫面的時間位置 | 停在輸入幀 | 能跟上真實幀 |

未來畫面應隨動作改變，所以打亂動作後，誤差應明顯上升；這表示最終模型確實有
使用動作資訊。模型／重複畫面的比值低於 1，表示預測優於直接複製輸入畫面。

在未參與訓練的操作片段上，以目標圖片規劃（左：選出的畫面；右：目標）：

![11 步抵達目標](gif/plan-roll-1.gif)

![4 步抵達目標](gif/plan-roll-2.gif)

![達到步數上限前走完 90 幀中的 71 幀](gif/plan-roll-3.gif)

## 為什麼調整論文設計

三項量測結果促成了這次調整：

1. **原始特徵模型忽略動作。** 打亂動作只讓誤差增加 0.2%；動作全設為零時也
   幾乎相同；連續預測則一直停在輸入畫面附近。
2. **編碼器仍保有動作資訊。** 一個 2,300 萬參數的探測模型可從兩幀的編碼
   特徵解出動作，R2 約為 0.37。
3. **只靠動作無法解釋原始特徵的變化。** Ridge regression 在不同場景間只能
   解釋 0.00% 的 latent delta 能量。模型必須同時理解動作與場景，但現有資料量
   不足以有效學會這種關係。

因此，專案加入以 inverse dynamics 訓練的 compressor。輔助模型必須從前後兩組
token 還原機器人動作，迫使壓縮後的特徵保留動態資訊，不讓靜態畫面內容主導
loss。

另外兩項選擇也來自量測：

- 每隔 6 幀取樣，因為原始 15 Hz 的單幀動作小於 encoder noise。
- 模型使用機器人實際產生、經角度修正的狀態變化，而不是要求它執行的指令。

## 系統設計

`固定的 V-JEPA encoder -> 動作感知 compressor -> causal predictor -> CEM 規劃器`

- **Encoder：** `facebook/vjepa2-vitl-fpc64-256` 把 256x256 畫面轉成
  256 x 1024 patch 特徵。每幀只編碼一次並存入快取；encoder 不參與訓練。
- **Compressor（約 7M）：** 16 個 learned query 從 256 個 patch 產生
  16 x 384 token。Inverse-dynamics head 讓 token 保留動作，輕量 reconstruction
  loss 則保留場景資訊。
- **Predictor（約 17M）：** 6 層 block-causal transformer，根據最多 16 幀及其
  動作預測下一組 token 的變化：`z[t+1] = z[t] + f(z[<=t], a[<=t])`。
- **動作輸入（7 維）：** 前 6 維是加總並修正角度的機器人狀態變化；最後一維
  是夾爪的絕對狀態。正規化數值會與 checkpoint 一起保存。

### 訓練方式

訓練分兩階段，先確認壓縮空間有效，再投入較昂貴的 predictor 訓練：

1. `train_compressor.py` 訓練 compressor、inverse-dynamics head，以及只在此階段
   使用的 reconstruction decoder。Motion R2 至少要達 0.2，compressed-space
   linear ceiling 至少要達 +2%，才繼續下一階段。
2. `train.py` 以較低 learning rate 微調 compressor，同時訓練 predictor。Loss
   包含 next-token prediction、two-pass rollout 與 inverse dynamics；另用
   stop-gradient target 與 motion monitor 防止 token collapse。

主要設定為 16 幀、stride 6，共涵蓋原始影片的 91 幀；依操作片段切分訓練／驗證
資料，並用 gradient accumulation 達到 batch size 64。

### 規劃 Demo

`plan_demo.py` 在沒有實體機器人的情況下示範 MPC。每一步會：

1. 用 CEM 產生多組 8 步動作；
2. 預測每組動作的結果，選出最後最接近目標 token 的一組；
3. 只執行最前面的幾個動作，把動作加到已記錄的機器人狀態，再選擇狀態最接近
   的已記錄畫面；
4. 把該真實畫面與實際位移加入上下文，重新規劃。

![依狀態選擇畫面](assets/snapping.svg)

已記錄的操作片段只是一個簡化環境，不能代表實體機器人控制已完成。執行過程
不讀取目標幀的時間位置，也不假設下一幀一定在影片的後方，以免自動「走向未來」
而製造成功結果。

常用控制：

- `--commit-steps`：當單一動作太小、無法切換到另一個已記錄狀態時，一次合併
  2-3 個動作。
- `--snap-range LO HI`：相同手臂姿勢出現在不同工作階段時，限制可選的畫面範圍。
  因為它使用時間範圍資訊，所以不是預設行為。
- `--action-momentum`：只參考上一個實際動作，減少規劃方向來回切換。

輸出會比較目標所需、模型要求、環境實際執行的動作。要求正確但執行錯誤，通常
代表錄影中缺少對應狀態；要求本身錯誤，則是規劃器的問題。

## 快速開始

```bash
uv sync                 # 測試與 CPU 快速檢查
uv sync --extra cache   # 建立快取、輸出 GIF 與 GPU 執行所需套件

uv run pytest
uv run scripts/train.py --model tiny --training smoke
```

依下方完整流程準備特徵快取後：

```bash
uv run scripts/evaluate.py       # 預設使用 weights/model.safetensors
uv run scripts/plan_demo.py      # 使用同一份 weights 與快取
```

可選路徑設定：`VJEPA_CACHE_DIR`（預設 `./latent_cache`）、`VJEPA_CKPT_DIR`
（`./checkpoints`）、`VJEPA_RECORDS_DIR`（`./records`）。

## 完整執行流程

請依序執行：

```bash
# 1. 下載資料，並建立各相機的 V-JEPA 特徵快取
uv run scripts/prepare_cache.py --episodes 100 --trim 15

# 2. 確認動作與狀態的意義
uv run scripts/check_actions.py --cache-dir latent_cache/wrist

# 3. 選擇相機與取樣間隔
uv run scripts/gate_sweep.py --seeds 1
uv run scripts/stride_gate.py --cache-dir latent_cache/ext1 --strides 4 6

# 4. 確認動作幾乎無法直接預測原始特徵變化
uv run scripts/ceiling_probe.py --stride 6

# 5. 訓練並驗證壓縮 token 空間
uv run scripts/train_compressor.py --stride 6

# 6. 訓練 action-conditioned predictor
uv run scripts/train.py --model base-c16 --training c-full --seed 0

# 7. 評估動作敏感度與連續預測品質
uv run scripts/evaluate.py

# 8. 執行目標圖片規劃
uv run scripts/plan_demo.py
```

`evaluate.py` 會與重複目前畫面、零動作、打亂動作三種 baseline 比較，並檢查預測
畫面在操作片段中的時間位置。通過標準為：最長預測距離下，打亂動作的誤差至少
增加 10%，且模型／重複畫面的誤差比不高於 0.9。

需要深入診斷時，可執行 `overfit_check.py --stride 6`。它在固定的 512 個 window
上比較正確動作與永久打亂動作的兩個 raw-feature model，用來區分「學得慢」與
「模型結構無法使用動作」。

## 設定與輸出

模型設定：

- `tiny`：在合成資料上執行 CPU 快速檢查。
- `tiny-c`：在 CPU 檢查 compressed-token 流程。
- `base`：raw-feature 失敗案例的 baseline。
- `base-c16`：最終的 16 x 384 compressor 與 predictor；需要第一階段 checkpoint。

訓練設定：

- `smoke`：stride 2、50 步合成資料。
- `full`：3,000 步 raw-feature baseline。
- `c-full`：stride 6、10,000 步 compressed-token 設定。

確切參數位於 `src/vjepa_ac/variations.py`。

主要輸出：

- `latent_cache/<camera>/`：快取特徵、狀態、動作與操作片段範圍。
- `checkpoints/<model>/<training>/<seed>/`：可繼續訓練與表現最佳的 checkpoint。
- `records/<model>/<training>/<seed>/record.jsonl`：訓練與評估數據。
- `records/diagnostics/`：相機、stride 與 ceiling 的診斷結果。
- `weights/model.safetensors`、`weights/model.json`：專案附帶的最終模型與設定；
  evaluation 和 planning 預設使用這組檔案。
