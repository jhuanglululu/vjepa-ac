# vjepa-ac

以學習為目的，複現 Meta V-JEPA 2 論文中的 V-JEPA 2-AC action-conditioned world model，並縮小到 100 段機器人 episode 與數個 GPU 小時所能負擔的規模：一個 frozen 的 V-JEPA encoder、一個以已執行動作為條件的小型 transformer predictor，以及使用 CEM 的 goal-image MPC planning。有趣的地方在於縮小規模所逼出來的取捨——model 是在一個 learned、以 motion 為權重的 16-token 空間中做預測，而非在 raw patch latent 上預測，而這個 deviation 的每一步都是由某項 measurement 所驅動的（詳見下文）。資料：`nvidia/Cosmos3-DROID` 的前 100 段 episode（success split，就意圖而言屬於單一 task family——相似的場景讓跨 episode 的 generalization 在這個規模下變得可行）。真正的執行跑在遠端 GPU 機器上；tests 與 smoke 變體則在本機 CPU 上跑。

在 held-out episode 上的 goal-image MPC——左：planner 所 commit 的畫面，右：給定的 goal（執行方式見「Demo 流程」）：

![朝 goal image 規劃，於 11 步抵達](gif/plan-roll-1.gif)

![第二段 episode，於 4 步抵達](gif/plan-roll-2.gif)

![第三次執行，於 step cap 時覆蓋了 +90 幀中的 +71 幀](gif/plan-roll-3.gif)

## 本專案與 Facebook 版的差異與原因

V-JEPA 2-AC 訓練一個約 300M 參數的 block-causal predictor，用來輸出下一幀的 **full patch latent**，資料為約 62 小時的 DROID，架在一個 frozen 的 ViT-g encoder 之上。在那樣的規模下，model 有本錢從 raw latent 中學到 action-x-content 的交互作用。而我們只有約 45 分鐘的機器人資料與一個 ViT-L/256 encoder，直接複現以一種具體、可 measure 的方式失敗了：

- Raw-latent 訓練是 **action-blind** 的：在 eval 時 shuffle 動作只讓誤差增加 +0.2%，zero-action 的結果與 model 一致，rollout 也從未離開輸入的那一幀（retrieval offset 剛好是 -h*stride）。rollout loss 並沒有幫助。
- 然而資訊確實存在：一個 23M 的 probe 能從 latent pair 中解出 motion，R2 ~0.37（stride_gate），所以失敗不在 encoder。
- 反方向才是致命的：從 action 對 raw latent delta 做 ridge regression，只解釋了它們 energy 的 **0.00%**（ceiling_probe）。在 V-JEPA latent space 中並不存在一個 scene-independent 的「moving right」方向——action 的使用必須經由 content-dependent 的交互作用，而 training loss 中約 99.9% 是 action 永遠無法解釋的 content/noise。一個 overfit 的 A/B test 顯示 action pathway 是以龜速打開的（在一個被背下來的 subset 上，3000 步內 sensitivity 從 0.7% 爬到 5%）——這是一個我們的資料預算無法花錢買通的 optimization 問題。

解法是改變 objective 而非放大 scale：**學一個小的 token space，讓 motion 在 variance 中佔有 first-class 的份額，然後在那裡做預測。** 一個 frozen-then-fine-tuned 的 compressor（16 個 cross-attention query，以 inverse dynamics ＋輕量 reconstruction 訓練）定義了這個空間；predictor 則完全在其中訓練。其餘沿用同一套論文 recipe——frozen encoder、block-causal predictor、Δstate conditioning、CEM planning——但 prediction target 是 16x384 的 token，而非 256x1024 的 latent。結果：在 h=15 時 shuffle 動作會讓誤差惡化 +50-65%（相對於 raw 版的 +0.2%），model/copy 為 0.67-0.73，rollout 能追上真實 trajectory。次要的 deviation 也全都由 measurement 驅動：以 temporal stride 6 取樣畫面（在 15 Hz 下每步的 motion 小於 encoder 的 noise floor；於 stride_gate 中量測），以及以 wrap-corrected 的**已執行（executed）** Δstate 作為 conditioning，而非以 commanded action。

## 架構

Frozen encoder -> compressor -> block-causal predictor，約 24M 個 trained param：

- **Encoder**（frozen，僅在建 cache 時使用）：`facebook/vjepa2-vitl-fpc64-256` 將每個 256x256 的畫面 map 成 256 個 patch x 1024 維。畫面只會被 encode 一次進入 latent cache；訓練過程從不碰 encoder。
- **Compressor C**（約 7M）：linear 1024->384 ＋逐 patch 的 MLP block，接著 16 個 learned query 對 256 個 patch 做 cross-attention（8 個 head），再接一個逐 token 的 MLP block -> 每幀 16 token x 384 維，並以 train-split 的統計量做 standardize，存為 model buffer。一個 **inverse-dynamics head**（僅 train-time）會從連續的 token pair 預測 conditioning feature——正是它把 motion 逼進 token 之中。
- **Predictor**（約 17M）：block-causal transformer（d_model 512、6 層、16 個 head、RoPE、SiLU MLP）。每幀貢獻它的 16 個 token 加上一個 action token（7 維 conditioning 的 linear embedding），因此一個 T=16 的 window 是一個 272-token 的序列；block-causal mask 讓第 t 幀的各位置能看到所有 <= t 的幀，包含 t 自己的 action token。output head 預測 residual token delta：z_{t+1} = z_t + f(z_<=t, a_<=t)。
- **Conditioning**：以每個 strided interval 為單位，dims 0-5 為在該 interval 內加總、經 wrap-correct 的 proprio Δstate，dim 6 為 absolute gripper，並以隨 checkpoint sidecar 一起傳遞的 train-episode 統計量做 normalize。

## 訓練流程

分兩個 phase，並設有 gate，讓 GPU 時間只花在某項正向 measurement 的下游：

1. **先訓練 compressor**（`train_compressor.py`，約數分鐘）：在 stride-6 的 latent pair 上訓練 C ＋ ID head——loss = inverse dynamics MSE ＋ 0.1 x reconstruction（一個用完即丟的 cross-attention decoder，regress 輸入的 patch，這讓 token 保有足夠的 static context 以供 forecasting）。以 held-out 的 ID motion R2 選出 best checkpoint。進入 phase 2 前必須通過兩道 gate：ID R2 >= 0.2，以及 C-space 的 ridge ceiling >= +2%（對應 raw-space 中那個 0.00% 的 measurement）。
2. **後訓練 predictor**（`train.py`，10k 步約 30 分鐘）：載入 phase-1 的 compressor，並以小得多的 lr（`compressor_lr`，約 lr/10）與 predictor 一起 fine-tune。loss = teacher-forced 的 smooth-L1（對下一批 token）＋一個 two-pass rollout 項（prediction 回饋一次）＋ ID auxiliary 項。解凍 compressor 會重新打開 collapse 的 pathway，因此有三道 guard 把它壓住：prediction target 做 stop-grad、ID auxiliary 讓 motion 能從 token linear 讀出、以及一個 collapse monitor 在每個 val interval 記錄 val token std ＋ ID loss。兩個 module 打包在同一個 checkpoint 裡。

訓練以 window 為單位：T=16 幀、stride 6（涵蓋 91 幀的跨度），episode-level 的 train/val split 由每支 script 共用，batch 64 並搭配 grad accumulation。

## Demo 流程

`plan_demo.py` 在沒有機器人的情況下跑論文的 goal-image planning loop，因此錄下來的 episode 必須充當環境。全程的設計準則是：planner 可以看 goal，但**執行不得參照 goal 或時間方向**——否則這個 demo 會自己製造出 success（在錄好的 trajectory 上做一個 time-forward 的 snap，依 construction 就能抵達任何未來的 goal）。

每個 committed step 的 loop 為：

1. **Plan**（忠於論文的 CEM）：在一個 8 步的 horizon 上，從每次 replan 都以 N(0,1) 重新初始化的 Gaussian 取樣 action sequence，從真實 context 出發把它們在 world model 中 roll 出來，以想像出的最終 token 與 goal 幀 token 之間的 L1 評分，對 elites 重新 refit，取其 mean。
2. **Execute** 只有前 `--commit-steps` 個 action，然後 **snap**：把 commanded 的、wrap-aware 的 Δstate 加到目前這幀的真實 proprio state 上，並 commit 狀態最接近（per-dim scaled、angle-wrapped）的那一段 episode 畫面。這就是 actuator；不會諮詢 world model。
3. **Re-ground**：被 commit 幀的真實 latent 加入 context（保留最後 4 個 real frame），並以被 commit 幀之間的**已執行（executed）** motion——而非 commanded action——作為 context 的 action row。重新 plan。

![state snapping](assets/snapping.svg)

各機制的用途：

- **State snapping** 解決 actuator 問題。model 自己 one-step 的 imagination 太過保守（約差 5 倍，因為 smooth-L1 regression 傾向於 mean），所以拿它當環境會 stall，更糟的是還會讓 model 自己考自己。以 ground-truth proprio 做 kinematics 則與 model 無關、任何方向都能運作（backwards commit 仍然可能，因此 failure 仍看得見），且與一個 tracking Δstate 指令的 position-controlled 機器人相對應。
- **`--commit-steps`** 解決 granularity 問題：一個 strided step 的 displacement 可能比相鄰兩幀之間的 state gap 還小，於是 snap 會回到同一幀而 stall；在 snap 前先 execute 2-3 個 planned action 即可跨過這個 spacing。
- **`--snap-range LO HI`** 解決 pose aliasing 問題。Proprio state 看不到世界，而一段 pour episode 會兩次經過幾乎相同的手臂姿態——一次是杯子在桌上、一次是杯子在 gripper 中——nearest-state snapping 會欣然地在這些 task phase 之間 teleport。把 snap pool 限縮到某個 frame range 可排除其他 phase；它輕微地 reference 了 goal 在時間上的位置，這也是為什麼它是一個 flag 而非 default。
- **`--action-momentum`** 解決 dithering 問題。忠於論文的 CEM 每次 replan 都從 zero mean 重啟，因此連續的 plan 可能彼此反向；以最後一個**已執行（executed）**的 action warm-start mean，會偏向延續真實的 motion。它只 reference 過去——絕不 reference goal 或時間方向——這正是讓它保持誠實之處（若改用 index-space momentum 就不誠實了）。
- **Executed-vs-commanded 的記帳**分離了 failure mode：header 印出所需的 start->goal motion，每一步都印出 commanded 與 executed 的 Δstate。commanded 與 required 一致但 executed 偏離，代表是 simulator 的問題（path 上沒有對應的幀）；commanded 與 required 不一致，則代表是 planner 的問題。

輸出：一份 per-step 的 trace、一份 committed frame 與 action 的 JSON，以及一支放在 checkpoint 旁的 side-by-side gif（committed 的 real frame vs goal image）。

## 環境設定

```
uv sync                 # local: tests + smoke runs (CPU is fine)
uv sync --extra cache   # remote: adds transformers/pyarrow/av/pillow for cache building + gifs
```

可選的環境變數（皆為路徑，附 default）：`VJEPA_CACHE_DIR`（`./latent_cache`）、`VJEPA_CKPT_DIR`（`./checkpoints`）、`VJEPA_RECORDS_DIR`（`./records`）。

## 各 script（依執行順序）

在做任何遠端動作前：`uv run pytest`（unit tests）與 `uv run scripts/train.py --model tiny --training smoke`（在 synthetic 資料上做 50 步的 CPU sanity check）都應在本機通過。那些你從不需要調整旋鈕的 script，只暴露 `--stride`/`--seed`；其餘一切都是 script 頂部的常數。

### 1. prepare_cache.py — 建立各 camera 的 latent cache

```
uv run scripts/prepare_cache.py --episodes 100 --trim 15
```

只下載涵蓋前 `--episodes` 段 episode 的 Cosmos3-DROID shard，丟掉每段的前 `--trim` 幀（手臂尚未進入視野），decode 每個 camera 的影片，並在最多 4 張空閒 GPU 上以 frozen 的 V-JEPA encoder encode 每一幀。為每個 camera 各寫一份 cache 到 `latent_cache/<cam>/`（`--cameras ext1 ext2 wrist`），內含 state、action 與 episode range，然後印出 health check。之後每支 script 都以 `--cache-dir` 或 `VJEPA_CACHE_DIR`（default `./latent_cache`）挑選 camera，所以選好 camera 後，你可以把那份 cache 移到根目錄並省略此 flag。

### 2. check_actions.py — 確認 state/action 語意

```
uv run scripts/check_actions.py --cache-dir latent_cache/wrist
```

印出 cache 中 commanded action、wrap-corrected 的 state delta 與 absolute state 之間的 per-dim 相關性。訓練前先確認：dims 0-5 表現得像 cartesian velocity 指令（corr(a,dS) 明顯為正），最後一維是 absolute gripper（corr(a,s) ~ +1）。

### 3. gate_sweep.py — 挑選 camera 與 stride

```
uv run scripts/gate_sweep.py --seeds 1     # quick pass, all cameras
uv run scripts/stride_gate.py --cache-dir latent_cache/ext1 --strides 4 6   # confirm winner, 3 seeds
```

gate_sweep 為每個 camera 各啟動一個 stride_gate，各自 pin 在一張空閒 GPU 上（至多 `--max-gpus 4`），並印出一張綜合 verdict 表。stride_gate 本身會針對每個 stride 與 seed，訓練兩個 23M 的 probe 以還原確切的 conditioning feature——一個來自 latent pair (z_t, z_{t+s})，一個是僅用 z0 的 control——因為與 z_t 冗餘的資訊拿來當 conditioning 是沒用的：decision statistic 是 pair 減 control 在 motion dim 上的 margin。誤差結合了對 held-out test episode 的 bootstrap 與 seed 之間的 spread，而一個 stride 只有在 pair R2 − SE 與 margin − SE 都跨過各自 threshold 時才通過。fail 會被標記為 conclusive 或 probe-limited（train R2 < 0.5），且 verdict 會檢查在所選 stride 下確實存在 training window。JSON 落在 `records/diagnostics/`。

### 4. ceiling_probe.py — 訓練前先估量 prize 大小

```
uv run scripts/ceiling_probe.py --stride 6
```

從 conditioning feature 對 raw latent delta 做 ridge regression，並回報 held-out、scene-independent、可歸因於 action 的那部分 training loss 份額。在這些 cache 上它讀到 ~0%：沒有 action-x-content 交互作用時，action 無法解釋任何 raw latent delta——正是這個 measurement 驅動了「在 learned 的 compressed space 中預測，而非在 raw latent 中預測」的決定。

### 5. train_compressor.py — phase 1：學習 prediction space

```
uv run scripts/train_compressor.py --stride 6
```

訓練 base-c16 compressor（16 個 learned query 對 256 個 patch 做 cross-attention）加上一個 inverse-dynamics head，並帶一個輕量 reconstruction 項，讓 token 保有 forecastable 的 context。以 held-out 的 ID motion R2 選出 best checkpoint，存到 `checkpoints/base-c16/comp-s<stride>/<seed>/compressor.safetensors`，並印出兩道 go/no-go gate：held-out ID R2 >= 0.2（compressor 找到了 motion signal）與 C-space linear ceiling >= +2%（此 token space 是由 action 驅動的，不像 raw latent）。若任一項 fail，就不要花 phase-2 的 GPU 時間。

### 6. train.py — phase 2：訓練 predictor

```
uv run scripts/train.py --model base-c16 --training c-full --seed 0
```

對 compressed model，它會自動載入 phase-1 的 compressor（可用 `--compressor` override），並以 `compressor_lr` fine-tune，帶三道 collapse guard：stop-grad target、inverse-dynamics auxiliary（`id_weight`），以及一個每個 val interval 印出 val token std ＋ ID loss 的 monitor（下降的 std 或上升的 ID loss 代表 compressor 在作弊——請調低 `compressor_lr`）。`--stride N` 會 override 該變體的 stride（record 存在 `<training>-s<N>`）；`--no-rollout` 會拿掉 two-pass rollout loss（`<training>-noroll`；在 stride 6 下 rollout loss 有可 measure 的幫助，因此 default 保留它）。若 run directory 已存在，會自動從 `current.safetensors` resume；checkpoint 會把 compressor ＋ predictor 打包在一起。

### 7. evaluate.py — action sensitivity 與 rollout 品質

```
uv run scripts/evaluate.py            # defaults to weights/model.safetensors
```

從 checkpoint sidecar 讀取 config ＋ conditioning stats，在 held-out episode 上將 model roll 出來，並依各 horizon 印出對 copy-first / zero-action / shuffled-action baseline 的 latent L1，以及 within-episode 的 frame retrieval。adoption criterion：在 max horizon 時 shuffled >= +10% 更差，且 model/copy <= 0.9。所交付的 weight 在 h=15 時得分為 shuffled +50-65%、model/copy 約 0.67-0.73，且 retrieval 能追上真實幀（median offset 約 0，相對於 copy 的 -h*stride）。

### 8. plan_demo.py — receding-horizon MPC demo

```
uv run scripts/plan_demo.py           # same default weights
```

挑一段 held-out episode，以 `--start` 的那幀作為 current state，並以往前 `--goal-offset` 幀（可為負）作為 goal image，接著 loop：CEM（每步以 N(0,1) 重新初始化）以對 goal token 的 L1 為想像出的 token rollout 評分，只有前 `--commit-steps` 個 action 會執行，且執行是與 model 無關的 kinematics——把 commanded 的、wrap-aware 的 dstate 施加到真實 proprio state 上，snap 到 nearest-state 的 episode 幀。context 只保留 real frame，並以它們之間的 executed（而非 commanded）motion。每步印出 required vs commanded vs executed 的 motion，以分離 planner error 與 simulator error；當 pose aliasing 跨 task phase teleport 時（相同手臂姿態、不同 world state），`--snap-range LO HI` 會把 snapping 限制在某個 frame window 內。存一支放在 checkpoint 旁的 side-by-side gif（committed frame vs goal）。

### overfit_check.py — 選用的 action-use 診斷

```
uv run scripts/overfit_check.py --stride 6
```

在一個固定的 512-window subset 上訓練一對 raw-space 的 twin model——true action vs 永久 shuffle——並回報 A/B 的 loss gap 加上 eval-time 的 shuffle sensitivity。當一個 model 忽略它的 action 時，可用來分離「optimization/scale 問題」（sensitivity 隨訓練增長）與「structural blindness」（維持在 zero 而不動）。

## 變體

**Model**
- `tiny` — smoke run 與 shape check，跑在一個小型 synthetic grid 上，絕不代表真實結果
- `tiny-c` — tiny 的 compressed-space twin，用來在本機演練 compressor path
- `base` — 論文規模的 predictor，用於 vjepa2-vitl 的 16x16x1024 latent grid（保留作為有記錄在案的 raw-latent negative baseline）
- `base-c16` — compressor（16 個 learned query 對 256 個 patch 做 query，以 inverse dynamics ＋輕量 reconstruction 訓練，再以 `compressor_lr` fine-tune）＋ 完全在 16x384 token space 中運作的 predictor；需要來自 train_compressor.py 的 phase-1 checkpoint

**Training**
- `smoke` — 在 synthetic 的 linear-dynamics 資料上做 50 步的本機 sanity check，stride 2
- `full` — 在真實 latent cache 上的 3k 步 recipe（raw 4112-token 序列）
- `c-full` — 給 compressed-space model 的 10k 步 stride-6 recipe（272-token 序列每步約便宜 15 倍）；`compressor_lr` 與 `id_weight` 住在這裡

僅為用途說明——數字住在 `src/vjepa_ac/variations.py`。新增變體 = 在那裡加一筆 entry，並在此處加一行，於同一次變更中完成。

## 檔案配置

- `records/<model>/<training>/<seed>/record.jsonl` — meta 行 ＋ per-step/eval metric
- `checkpoints/<model>/<training>/<seed>/` — `<step>.safetensors`（＋ `<step>.json` sidecar）、供 resume 的 `current.*`，依 val loss 保留 3 個 best
- `weights/` — `model.safetensors` ＋ `model.json`：已訓練的 seed-0 `base-c16`/`c-full` model（weight ＋ sidecar，已 commit 進 repo）；evaluate.py 與 plan_demo.py 都以它為 default，因此兩者不需任何 checkpoint path 即可執行
- `latent_cache/<cam>/` — 來自 prepare_cache 的每 camera `latents.safetensors` ＋ `cache.json`
- `records/diagnostics/` — stride_gate/gate_sweep 的輸出
