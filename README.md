# llm_jepa_v2

1. This repo uses poetry.
2. `train.py` is the main entrypoint

## Run with Google Compute Engine + Docker

This project uses Docker; the `Dockerfile` handles all dependency management and setup.

### Build the Image

From the project root (where the Dockerfile lives):

(For CPU (NOT RECOMMENDED: Compute is too intenstive for CPU, but can be used to test dependencies))

```bash
docker build -f Dockerfile.cpu -t myapp:cpu .
docker run --rm myapp:cpu
```

(For GPU (RECOMMENDED))

```bash
docker build -f Dockerfile.gpu -t myapp:gpu .
docker run --rm --gpus all myapp:gpu
```

Then run

```bash
./submit_job_to_gcp_compute.sh
```

-v "$PWD":/app mounts your current directory so code changes are visible inside the container.
--gpus all exposes all available GPUs to the container (omit for CPU).



## Converting the docker container to Apptainer for Gadi

Gadi runs **Apptainer (Singularity)**, not Docker.  
Workflow: **Build Docker → Convert to `.sif` → Upload to Gadi → (optional) Run with PBS**

### 1️⃣ Build the Docker image
```bash
docker build -f Dockerfile.gpu -t myapp:latest .
```

### 2️⃣ Convert to an Apptainer .sif

```
mkdir -p sif_out
docker save myapp:latest -o myapp-oci.tar
docker run --rm -it \
  -v "${PWD}:/work" -w /work \
  -v "${PWD}/sif_out:/out" \
  ghcr.io/apptainer/apptainer:latest \
  apptainer build /out/myapp.sif docker-archive:///work/myapp-oci.tar
```

### 3️⃣ Upload to Gadi
```
scp sif_out/myapp.sif <NCI_USERNAME>@gadi.nci.org.au:/g/data/<project>/containers/
```

### 4️⃣ Test on Gadi
```
module load apptainer
apptainer exec /g/data/<project>/containers/myapp.sif python --version
apptainer exec --nv /g/data/<project>/containers/myapp.sif nvidia-smi  # GPU check
```

### 5️⃣ Optional PBS script (2 × A100 GPUs)

```
#!/bin/bash
#PBS -P <project>
#PBS -q dgxa100
#PBS -l ncpus=16,ngpus=2,mem=64GB,jobfs=200GB,walltime=04:00:00
#PBS -l storage=gdata/<project>+scratch/<project>
#PBS -N myapp-train
#PBS -j oe

module load apptainer
IMG=/g/data/<project>/containers/myapp.sif
WORKDIR=$PBS_JOBFS/work
mkdir -p "$WORKDIR" && cd "$WORKDIR"

apptainer exec --nv \
  --bind /g/data/<project>:/gdata \
  --bind /scratch/<project>:/scratch \
  "$IMG" python /app/train.py
```