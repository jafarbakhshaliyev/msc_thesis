#!/bin/bash
#SBATCH --job-name=face_detection
#SBATCH --output=fd.out
#SBATCH --error=fd.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40 
#SBATCH --chdir=/home/bakhshaliyev/classification-aug/MultiRocket # change to the directory


echo "Activating environment..."
source ~/venvs/sktime-env/bin/activate # change to your virtual environment path

echo "Running script..."


srun python3 main.py --problem FaceDetection --iter 5 --verbose 1

srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --jitter
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --scaling
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --rotation
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --permutation
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --randompermutation
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --magwarp
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --timewarp
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --windowslice
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --windowwarp
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --spawner
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --dtwwarp
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --shapedtwwarp
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --wdba
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --discdtw
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --discsdtw

srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 420 --stride 1 --shuffle_rate 0.8 
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 120 --stride 1 --shuffle_rate 0.7
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 4 --stride 4 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 64 --stride 2 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 32 --stride 1 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 8 --stride 4 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 6 --stride 4 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 120 --stride 1 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 96 --stride 8 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 48 --stride 36 --shuffle_rate 0.9
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 24 --stride 36 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 240 --stride 1 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 240 --stride 1 --shuffle_rate 0.4
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 240 --stride 1 --shuffle_rate 0.2
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 96 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 96 --stride 96 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 120 --stride 48 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 240 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 12 --stride 96 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 2 --stride 1 --shuffle_rate 0.4
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 480 --stride 1 --shuffle_rate 0.7
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 900 --stride 1 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 64 --stride 24 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 64 --stride 24 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 148 --stride 24 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 148 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 148 --stride 48 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 180 --stride 12 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 180 --stride 24 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 180 --stride 12 --shuffle_rate 0.5
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 152 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 360 --stride 1 --shuffle_rate 0.1
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 72 --stride 2 --shuffle_rate 0.4
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tips --patch_len 120 --stride 1 --shuffle_rate 0.2

srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 420 --stride 1 --shuffle_rate 0.8 
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 120 --stride 1 --shuffle_rate 0.7
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 4 --stride 4 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 64 --stride 2 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 32 --stride 1 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 8 --stride 4 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 6 --stride 4 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 120 --stride 1 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 96 --stride 8 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 48 --stride 36 --shuffle_rate 0.9
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 24 --stride 36 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 240 --stride 1 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 240 --stride 1 --shuffle_rate 0.4
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 240 --stride 1 --shuffle_rate 0.2
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 96 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 96 --stride 96 --shuffle_rate 0.6
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 120 --stride 48 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 240 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 12 --stride 96 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 2 --stride 1 --shuffle_rate 0.4
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 480 --stride 1 --shuffle_rate 0.7
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 900 --stride 1 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 64 --stride 24 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 64 --stride 24 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 148 --stride 24 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 148 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 148 --stride 48 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 180 --stride 12 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 180 --stride 24 --shuffle_rate 0.8
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 180 --stride 12 --shuffle_rate 0.5
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 152 --stride 96 --shuffle_rate 1.0
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 360 --stride 1 --shuffle_rate 0.1
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 72 --stride 2 --shuffle_rate 0.4
srun python3 main.py --problem FaceDetection --iter 5 --verbose 1 --use-augmentation --augmentation-ratio 1 --tps --patch_len 120 --stride 1 --shuffle_rate 0.2




echo "Job finished"
