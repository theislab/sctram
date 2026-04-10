# --- config (re-use your password file) ---
password_file="/Users/kemalinecik/Documents/Helmholtz/password.txt"
REMOTE_USER="kemal.inecik"
REMOTE_HOST="hpc-build01"
REMOTE_DIR="/home/icb/kemal.inecik/lustre_workspace/temp_move"
LOCAL_BASE="$HOME/Downloads"

# --- make a unique destination folder under Downloads ---
timestamp="$(date +"%Y%m%d_%H%M%S")"
TARGET_DIR="$LOCAL_BASE/temp_move_${timestamp}"
mkdir -p "$TARGET_DIR"

# --- rsync over sshpass (progress + resume, excluding one file) ---
RSYNC_SSH="sshpass -f \"$password_file\" ssh -o LogLevel=error"

rsync -avh \
  --partial --inplace \
  --progress --stats \
  --exclude "unification_union_20240330_hvg-intersection_integration.h5ad" \
  -e "$RSYNC_SSH" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" \
  "$TARGET_DIR/"

# --- optional: quick summary after copy ---
echo "Copied to: $TARGET_DIR"
du -sh "$TARGET_DIR"
find "$TARGET_DIR" -type f | wc -l | awk '{print "Files:", $1}'
