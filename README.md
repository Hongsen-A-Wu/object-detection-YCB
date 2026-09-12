python3 -m pip install -U "huggingface_hub[cli]"

hf download \
  bop-benchmark/ycbv \
  ycbv_base.zip \
  --repo-type dataset \
  --local-dir .

hf download \
  bop-benchmark/ycbv \
  ycbv_models.zip \
  --repo-type dataset \
  --local-dir .

hf download \
  bop-benchmark/ycbv \
  ycbv_test_all.zip \
  --repo-type dataset \
  --local-dir .

hf download \
  bop-benchmark/ycbv \
  ycbv_test_bop19.zip \
  --repo-type dataset \
  --local-dir .

hf download \
  bop-benchmark/ycbv \
  ycbv_train_pbr.zip \
  --repo-type dataset \
  --local-dir .

hf download \
  bop-benchmark/ycbv \
  ycbv_train_real.z01 \
  --repo-type dataset \
  --local-dir .

hf download \
  bop-benchmark/ycbv \
  ycbv_train_real.zip \
  --repo-type dataset \
  --local-dir .

hf download \
  bop-benchmark/ycbv \
  ycbv_train_synt.zip \
  --repo-type dataset \
  --local-dir .


7zz x ycbv_train_real.zip