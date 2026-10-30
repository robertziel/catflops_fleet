# shellcheck shell=bash
# WHERE CHECKPOINTS LIVE, for every deploy script — from the environment, else lb/.env, never a hardcoded folder
# (operator, 2026-10-09: "models dir also should be in .env defined, we use HDD for that"):
#   MODELS_DIR          where checkpoints are downloaded to and kept — this deployment's HDD, /srv/models
#   MODELS_SEARCH_DIRS  where a box looks for one, best first, `:`-separated, `~` the BOX's home; the first entry is
#                       where `stage` copies to (the fast local disk)
# The LB reads the same MODELS_SEARCH_DIRS from lb/.env (vllm_lb.catalog). Unset is an error naming the variable.
# MODELS_ENV_FILE names another env file to read (the deploy tests point it at nothing).
_models_env_file=${MODELS_ENV_FILE-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lb/.env"}
for _models_var in MODELS_DIR MODELS_SEARCH_DIRS; do
  if [[ -z ${!_models_var:-} && -r $_models_env_file ]]; then
    _models_val=$(sed -n "s/^${_models_var}=//p" "$_models_env_file" | tail -n 1)
    _models_val=${_models_val%\"}; _models_val=${_models_val#\"}
    [[ -n $_models_val ]] && printf -v "$_models_var" '%s' "$_models_val"
  fi
done
unset _models_var _models_val _models_env_file
: "${MODELS_DIR:?MODELS_DIR is not set — the folder checkpoints are downloaded to (lb/.env)}"
: "${MODELS_SEARCH_DIRS:?MODELS_SEARCH_DIRS is not set — where each box looks for a checkpoint, best first (lb/.env)}"

# A search dir as the REMOTE shell must see it: a leading `~` becomes $HOME, which only the box can expand.
remote_dir() { local d=$1; [[ $d == "~"* ]] && d="\$HOME${d#\~}"; printf '%s' "$d"; }
