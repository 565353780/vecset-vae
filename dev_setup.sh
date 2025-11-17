cd ..
git clone git@github.com:565353780/base-trainer.git

cd base-trainer
./dev_setup.sh

pip install omegaconf jaxtyping typeguard mcubes \
  scikit-image diso opencv-python

pip install flash-attn --no-build-isolation
