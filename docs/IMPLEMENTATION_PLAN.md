# Implementation plan: leaderboard + paper

Ngày khóa kế hoạch: 2026-10-23  
Mục tiêu: tối đa hóa Panoptic Quality (PQ) một cách hợp lệ, đồng thời tạo một nghiên cứu có protocol, ablation và mã nguồn tái lập được.

## 1. Kết luận về chiến lược hiện tại

Hướng `instance detector + high-resolution semantic refiner + instance-guided fusion` là đúng và nên là trục chính. Tuy nhiên, cần sửa thứ tự ưu tiên:

1. Audit và tái lập notebook YOLO + U-Net công khai đang hiển thị public score 0.69 trước khi tự viết toàn bộ pipeline.
2. Khóa evaluator, split và tracking experiment trước khi train thêm.
3. Phân tích FN/FP/mask-IoU để biết bottleneck là detection hay mask refinement.
4. Cải thiện detector hiện tại và checkpoint selection.
5. Xây refiner và fusion tối thiểu; chỉ thêm topology loss khi BCE + Dice đã chạy ổn.
6. Chỉ thử Mask DINO nếu detector vẫn là bottleneck sau fusion.
7. Ensemble sau cùng, không dùng để che một pipeline đơn yếu.

Không đặt cược toàn bộ vào `YOLO 2048`. Tăng resolution là ablation rẻ và hợp lý, nhưng không bảo đảm giải quyết FN, merge/split hoặc bất đồng annotator.

## 2. Hai protocol phải tách biệt

### Track L — Leaderboard

- Dùng một development fold cố định để ra quyết định nhanh.
- Sau khi khóa hyperparameters, train lại bằng toàn bộ official training data.
- Dùng Kaggle public score như tín hiệu phụ, không tune liên tục theo leaderboard.
- Không dùng annotations/spines của test images từ bản MAGFiLO công khai bên ngoài competition.

### Track P — Paper

- Group theo physical `file_name`; tuyệt đối không để các annotator của cùng ảnh rơi vào hai fold.
- Fold 0 là development fold để chọn kiến trúc và hyperparameters.
- Sau khi khóa config, chạy các fold còn lại không chỉnh tham số nữa.
- Báo cáo mean, standard deviation và bootstrap 95% CI theo physical image.
- Báo cáo Kaggle score riêng; không trình bày leaderboard như bằng chứng tổng quát hóa chính.

## 3. Cấu trúc repo đích

```text
configs/
  data.yaml
  yolo_1536.yaml
  yolo_2048_ft.yaml
  refiner_bce_dice.yaml
  refiner_boundary.yaml
  refiner_skeleton.yaml
  fusion_v1.yaml
  ensemble_v1.yaml
docs/
  IMPLEMENTATION_PLAN.md
  EXPERIMENT_LOG.md
src/filament/
  data/
    build_splits.py
    annotators.py
    crops.py
  metrics/
    pq.py
    stratified.py
  models/
    yolo.py
    refiner.py
  analysis/
    error_taxonomy.py
    visualize_errors.py
  fusion.py
  ensemble.py
  registry.py
  submit.py
tests/
  test_pq.py
  test_rle.py
  test_split_leakage.py
  test_non_overlap.py
```

Mọi artifact server nằm dưới `${ARTIFACT_ROOT}`:

```text
data/       split manifests và cache
runs/       checkpoint từng experiment
predictions/raw/
predictions/fused/
results/    metrics, CSV và submission
reports/    plots, error gallery, tables
logs/
```

Mỗi run phải lưu `config.yaml`, `split.json`, seed, git commit, package versions, checkpoint, per-image metrics và runtime. Không dùng một `run_name` hard-code cho nhiều thí nghiệm.

## 4. Roadmap triển khai

### P0 — Reproducibility và safety rails (23/10)

#### P0.1 Refactor CLI/config

- Thêm `--run-name`, `--checkpoint`, `--split-manifest`, `--config`, `--seed` cho train/evaluate/infer.
- Không mặc định mọi lệnh vào `yolov8l_seg_grouped_baseline`.
- `infer` phải ghi metadata checkpoint và postprocess parameters cạnh submission.
- Tạo `registry.csv` hoặc `registry.jsonl` tự động sau mỗi evaluate.

Acceptance criteria:

- Hai run khác tên không ghi đè nhau.
- Có thể evaluate checkpoint bất kỳ bằng đường dẫn tường minh.
- Từ một dòng registry truy ra được config, split và submission tương ứng.

#### P0.2 Khóa split

- Tạo 5 folds theo physical filename.
- Stratify xấp xỉ theo năm/observatory suffix, số instance và tổng foreground area.
- Lưu manifest bất biến; ghi hash vào mọi run.
- Unit test bảo đảm một physical image chỉ thuộc một fold.

#### P0.3 Khóa official PQ

- Bọc logic của official self-evaluation notebook thành module/test.
- Tạo synthetic tests cho TP, FP, FN, strict IoU threshold, empty image và mask overlap.
- So sánh output module với notebook trên cùng một prediction cache.

Acceptance criteria: PQ/TP/FP/FN trùng official evaluator đến sai số số thực cho bộ kiểm thử và validation hiện tại.

### P1 — Audit baseline và notebook 0.69 (23–24/10)

#### P1.1 Freeze baseline hiện tại

Ghi vào registry:

- YOLOv8l-seg, 1536, 60 epochs.
- Local PQ 0.353937 tại `conf=0.20`, `min_area=200`.
- TP 843, FP 519, FN 750.
- Kaggle score 0.33.

Copy checkpoint/config/result metadata thành một experiment bất biến; không train đè lên nó.

#### P1.2 Reproduce/audit public YOLO + U-Net notebook

- Tải source notebook và tất cả dependency công khai.
- Xác định chính xác weights được train ở đâu, labels nào được dùng và preprocessing/fusion nào tạo score.
- Kiểm tra không dùng test GT, hand labels hoặc inaccessible private artifact.
- Chạy inference trên một tập nhỏ, sau đó full validation/test nếu hợp lệ.
- Tách code hữu ích thành module; không giữ logic cốt lõi chỉ trong notebook.

Go/no-go:

- Nếu hợp lệ và tái lập được: dùng làm strong reference baseline và ưu tiên port pipeline.
- Nếu thiếu private weights hoặc có leakage: chỉ học ý tưởng, không dùng score 0.69 làm mốc khoa học.

#### P1.3 Error taxonomy

Với mỗi unmatched GT/prediction, phân loại:

1. Missed detection: không có predicted box/mask gần GT.
2. Localization/mask failure: có candidate nhưng mask IoU không vượt 0.5.
3. Split: một GT tương ứng nhiều predictions.
4. Merge: nhiều GT tương ứng một prediction.
5. False dark structure: sunspot/artifact/background.
6. Annotation disagreement: annotators không thống nhất instance/extent.

Xuất `per_image_metrics.csv` và gallery 25 ảnh xấu nhất/từng nhóm lỗi.

Decision rule:

- Missed detection chiếm đa số: ưu tiên detector, resolution, sampling.
- Candidate đúng nhưng IoU thấp: ưu tiên refiner.
- Split/merge cao: ưu tiên fusion/instance assignment.
- FP từ sunspots cao: hard-negative mining và disk/radial features.

### P2 — Detector experiments (24–27/10)

Chạy trên cùng fold 0 và cùng evaluator. Không đổi nhiều yếu tố trong một experiment.

| ID | Experiment | Thay đổi |
|---|---|---|
| D0 | Reproduce baseline | YOLOv8l-seg 1536 |
| D1 | PQ checkpoint sweep | Evaluate mọi epoch; full grid chỉ cho top 3 |
| D2 | 2048 fine-tune | Start từ D1; 20–30 epochs |
| D3 | Conservative augment | no mosaic/mixup/translate; flips/90° rotation; scale nhỏ |
| D4 | Annotator medoid | một instance target set đại diện/physical image |
| D5 | Random annotator | sample một annotation set mỗi epoch |
| D6 | YOLO11m/l-seg | chỉ khi D2–D5 đã rõ bottleneck |

Khởi tạo D2:

```yaml
imgsz: 2048
epochs: 30
mosaic: 0.0
mixup: 0.0
translate: 0.0
scale: 0.10
fliplr: 0.5
flipud: 0.5
amp: true
```

Learning rate và batch size được chọn bằng short smoke run; không giả định config trên là tối ưu.

Promotion gate:

- Chỉ promote nếu PQ fold 0 tăng tối thiểu 0.01 tuyệt đối hoặc giảm rõ FN/mask failure đúng bottleneck.
- Inspect ít nhất 30 ảnh; không promote một run chỉ vì grid postprocess overfit.
- Giữ top 2 detector có lỗi bổ sung nhau cho ensemble/fusion.

### P3 — High-resolution structure refiner (26–31/10)

#### P3.1 Dataset/crop generator

- Crop từ GT boxes khi train; expand box 15–30% để có context.
- 50% positive crops, 25% near-miss/hard dark negatives, 25% background negatives.
- Native-resolution crop 512 hoặc 768; resize chỉ khi ROI lớn hơn crop budget.
- Input ban đầu: 3 channels `[normalized gray, CLAHE/local contrast, detector seed mask]`.
- Với nhiều annotators: semantic soft target là tần suất pixel được annotate; instance target của detector vẫn được xử lý riêng.

#### P3.2 Refiner baselines

| ID | Model/loss | Mục đích |
|---|---|---|
| R0 | ResNet34 U-Net, BCE + Dice | baseline tối giản |
| R1 | U-Net++/EfficientNet-B3, BCE + Dice | tăng capacity |
| R2 | R1 + boundary term | kiểm tra boundary quality |
| R3 | R1 + Skeleton Recall hoặc clDice | topology ablation |

Không train R2/R3 trước khi R0/R1 và crop reconstruction đúng.

Metrics riêng của refiner:

- Dice/IoU trong ROI.
- Boundary F-score.
- Skeleton recall/clDice.
- Quan trọng nhất: PQ sau fusion, split count và merge count.

Topology loss chỉ được giữ nếu PQ tăng mà merge không xấu đi. Literature gần đây cho thấy tăng skeleton coverage có thể đánh đổi precision/region overlap, nên đây là ablation chứ không phải mặc định.

### P4 — Instance-guided fusion (29/10–03/11)

#### Fusion v1

1. Detector tạo box, confidence, coarse mask/seed.
2. Expand từng ROI và chạy refiner ở native resolution.
3. Threshold probability trong ROI.
4. Chỉ giữ connected component chạm detector seed; cho phép controlled gap repair nhỏ.
5. Pixels thuộc nhiều proposals được gán theo `refiner_prob × detector_conf × seed_affinity`.
6. Sau assignment, masks phải rời nhau tuyệt đối trước RLE.
7. Loại mask theo calibrated confidence/area/shape, không chỉ một `min_area` toàn cục.

#### Fusion v2 nếu merge/split vẫn cao

- Geodesic/watershed assignment từ các detector seeds trên refiner probability map.
- Merge proposals chỉ khi mask IoU/support từ nhiều views cao.
- Không dùng morphology closing lớn trên toàn ảnh vì dễ nối hai filament khác nhau.

Ablations bắt buộc:

- detector-only;
- refiner connected-components-only;
- naive intersection;
- naive union;
- instance-guided fusion v1;
- fusion v2 nếu có.

Tune trên cache để không rerun GPU inference cho mỗi postprocess setting. Grid thô trước, search nhỏ quanh vùng tốt nhất sau.

Promotion gate: hybrid phải tăng PQ ít nhất 0.015 so với detector tốt nhất và gain xuất hiện trên nhiều size bins, không chỉ vài ảnh lớn.

### P5 — Paper-oriented annotator study (01–06/11)

So sánh trên cùng backbone/fold:

- duplicate annotation records như baseline hiện tại;
- medoid annotator target;
- random annotator per epoch;
- soft consensus cho semantic refiner;
- uncertainty-aware weighting: giảm trọng số pixel có disagreement cao.

Phân tích gain theo annotator agreement. Contribution tiềm năng chỉ được khẳng định nếu hiệu quả lặp lại trên confirmatory folds.

### P6 — Advanced detector/ensemble (04–09/11)

#### Mask DINO gate

Chỉ bắt đầu nếu error taxonomy sau P4 vẫn cho thấy detector miss là bottleneck lớn và còn compute budget. Dùng R50 pretrained, crop/resize policy kiểm soát, trước tiên chạy một fold. Dừng nếu không vượt detector tốt nhất hoặc không cung cấp predictions bổ sung đáng kể.

#### TTA

- 4-view flips/rotations phù hợp vật lý ảnh tròn.
- Invert transform chính xác rồi match instances.
- Giữ TTA chỉ khi PQ tăng đủ bù runtime.

#### Ensemble

- Cluster instances giữa models/views bằng mask IoU và box overlap.
- Weighted mask probability/majority support trong cluster.
- Singleton phải qua calibrated confidence + area/shape gates.
- De-overlap cuối cùng bằng score map, không greedy area đơn thuần nếu có thể.

Ensemble chỉ dùng 2–3 members có lỗi bổ sung nhau. Không ensemble nhiều checkpoint gần giống nhau mà không tăng OOF PQ.

### P7 — Confirmatory evaluation và final training (08–12/11)

- Khóa code/config trước confirmatory folds.
- Chạy config thắng trên các folds còn lại; không đổi threshold sau khi xem kết quả.
- Báo cáo PQ, SQ, RQ, TP/FP/FN, precision/recall, Dice/IoU distribution, split/merge rates.
- Breakdown theo instance area, skeleton length/thickness, observation source/year và annotator agreement.
- Chọn final config dựa trên mean confirmatory PQ + ổn định, không chỉ fold tốt nhất.
- Train final models trên toàn bộ official train data.
- Sinh 2 candidate submissions: best single/hybrid và ensemble.
- Chạy submission audit: schema, RLE decode, image IDs, bounds, empty images, zero overlaps, duplicate masks.

### P8 — Report/release (10–15/11)

Paper/report title làm việc:

> Instance-Guided High-Resolution Refinement for Solar Filament Segmentation under Annotator Disagreement

Research questions:

- RQ1: Native-resolution refinement cải thiện PQ ở filament mảnh/nhỏ đến đâu?
- RQ2: Instance-guided fusion giảm mask-IoU failures mà không tăng merge như thế nào?
- RQ3: Medoid/random/soft-consensus supervision xử lý multi-annotator disagreement ra sao?
- RQ4: Boundary/topology losses có thực sự tăng instance-level PQ hay chỉ tăng structural recall?

Figures/tables bắt buộc:

1. Sơ đồ detector → refiner → fusion.
2. Dataset/split/annotator statistics.
3. Main ablation table với compute cost.
4. PQ decomposition và size/thickness breakdown.
5. Error taxonomy trước/sau fusion.
6. Qualitative TP, FP, FN, split, merge và disagreement cases.
7. Local CV và Kaggle score trong hai cột tách biệt.

Release checklist:

- README chạy được từ clean environment.
- requirements/lock file và hardware/runtime.
- split manifests, configs, seeds và checkpoint provenance.
- training, evaluate, infer và submission commands.
- license/attribution cho mọi notebook, code và pretrained weights.
- không phát hành competition data hoặc artifact không có quyền redistribute.

## 5. Experiment ledger tối thiểu

Mỗi dòng phải có:

```text
experiment_id
date
git_commit
data_hash
split_hash
model
initial_checkpoint
config_path
seed
gpu/runtime
checkpoint_path
PQ/SQ/RQ/TP/FP/FN
per-size metrics
postprocess parameters
Kaggle score (nếu submit)
decision: reject/promote
notes
```

Không so sánh các con số validation từ repository khác như cùng protocol. Chỉ dùng chúng để tạo hypothesis.

## 6. Submission discipline

- Local PQ là gate bắt buộc trước mọi Kaggle submission.
- Mỗi submission kiểm tra đúng checkpoint/config và lưu SHA-256 của CSV.
- Không submit hai biến thể chỉ khác threshold rất nhỏ nếu local evidence không rõ.
- Ghi public score nhưng không chọn mô hình chỉ bằng public leaderboard.
- Giữ submission cuối cho best single/hybrid và best ensemble có OOF evidence.

## 7. Milestones và stop rules

| Milestone | Deliverable | Gate |
|---|---|---|
| M0 | Evaluator/split/registry tin cậy | tests pass, baseline reproduced |
| M1 | Detector tốt nhất | +0.01 PQ hoặc lỗi mục tiêu giảm rõ |
| M2 | Refiner + fusion | +0.015 PQ trên detector tốt nhất |
| M3 | Confirmatory folds | gain nhất quán, CI và error analysis |
| M4 | Final submission/report | reproducible, no leakage, audit pass |

Stop rules:

- Dừng high-resolution scaling nếu hai run không tăng PQ/miss recall.
- Dừng topology loss nếu merge/FP tăng hoặc PQ không tăng.
- Dừng Mask DINO sau một fold nếu không có gain hay complementarity.
- Dừng ensemble member nếu ablation bỏ member không làm PQ giảm.

## 8. Thứ tự triển khai ngay

1. P0.1: refactor run/checkpoint/config paths.
2. P0.2–P0.3: split manifest + evaluator tests.
3. P1.2: tải và audit notebook YOLO + U-Net 0.69.
4. P1.3: tạo error taxonomy cho checkpoint baseline hiện tại.
5. D1: sweep saved checkpoints theo PQ trên prediction cache.
6. D2/D3: YOLO 2048 fine-tune và conservative augmentation.
7. R0 + fusion v1 tối thiểu.
8. Chỉ sau đó mới mở R1–R3, Mask DINO và ensemble.

Top 1 không thể được bảo đảm, đặc biệt khi public leaderboard có thể chứa nghiệm không hợp lệ hoặc không tái lập. Mục tiêu thực tế là xây pipeline hợp lệ vượt strong open baselines, có local evidence ổn định, và đủ sạch để qua source-code/technical-report review.
