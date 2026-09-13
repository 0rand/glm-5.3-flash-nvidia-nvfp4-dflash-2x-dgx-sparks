#!/bin/bash
#
# Head-driven hybrid launcher for vLLM Docker clusters.
#
# Managed Ray mode (default when --workers and a command after -- are supplied):
#   - run ONLY on the head node
#   - launch idle Docker containers on head and workers
#   - start the Ray head in the head container
#   - start Ray workers over passwordless SSH
#   - execute the vLLM command once in the head container
#
# No-Ray mode (--no-ray):
#   - run ONLY on the head node
#   - launch idle Docker containers on all nodes
#   - start vLLM worker ranks first, then rank 0 on the head
#
# Legacy Ray mode remains available for compatibility:
#   - if --workers and a command after -- are NOT supplied,
#     run this script separately on each node with --head or --worker.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  Managed Ray mode (head node only):
    bash run_cluster_dual.sh <docker_image> <head_node_ip> --head <path_to_hf_home> \
      --workers <worker1_ip,worker2_ip,...> [--master-port <ray_head_port>] \
      [--container-name <name>] [docker args...] -- <vllm command>

  No-Ray mode (head node only):
    bash run_cluster_dual.sh <docker_image> <head_node_ip> --head <path_to_hf_home> \
      --no-ray --workers <worker1_ip,worker2_ip,...> [--master-port <port>] \
      [--container-name <name>] [docker args...] -- <vllm command>

  Legacy Ray mode (run separately on each node):
    bash run_cluster_dual.sh <docker_image> <head_node_ip> --head|--worker \
      <path_to_hf_home> [docker args...]

Notes:
  * Managed Ray and No-Ray modes are launched only from the head node.
  * The head must have passwordless SSH access to every worker.
  * <path_to_hf_home> must exist at the same host path on every node.
  * In Managed Ray mode, --master-port is the Ray head port.
  * No Ray object-store or CPU limits are forced; Ray defaults are used.
USAGE
}

if [[ $# -lt 4 ]]; then
    usage
    exit 1
fi

DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"
PATH_TO_HF_HOME="$4"
shift 4

MODE="ray"
MASTER_PORT="29501"
WORKERS_CSV=""
CONTAINER_NAME="node-${RANDOM}"
CLUSTER_COMMAND=()
DOCKER_ARGS=()
BASE_DOCKER_ARGS=()
WORKER_NODES=()
VLLM_HOST_IP=""
HEAD_DRIVEN_ACTIVE="false"

if [[ "$NODE_TYPE" != "--head" && "$NODE_TYPE" != "--worker" ]]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

shell_join() {
    local out=""
    local arg q
    for arg in "$@"; do
        printf -v q '%q' "$arg"
        out+="$q "
    done
    printf '%s' "$out"
}

extract_vllm_host_ip() {
    local args=("$@")
    local i arg next
    for ((i = 0; i < ${#args[@]}; i++)); do
        arg="${args[$i]}"
        case "$arg" in
            -e)
                next="${args[$((i + 1))]:-}"
                if [[ "$next" == VLLM_HOST_IP=* ]]; then
                    printf '%s' "${next#VLLM_HOST_IP=}"
                    return 0
                fi
                ;;
            -eVLLM_HOST_IP=*|VLLM_HOST_IP=*)
                printf '%s' "${arg#*=}"
                return 0
                ;;
        esac
    done
    return 1
}

strip_vllm_host_ip_env() {
    local args=("$@")
    local cleaned=()
    local i arg next
    for ((i = 0; i < ${#args[@]}; i++)); do
        arg="${args[$i]}"
        case "$arg" in
            -e)
                next="${args[$((i + 1))]:-}"
                if [[ "$next" == VLLM_HOST_IP=* ]]; then
                    ((i++))
                    continue
                fi
                cleaned+=("$arg")
                ;;
            -eVLLM_HOST_IP=*|VLLM_HOST_IP=*)
                continue
                ;;
            *)
                cleaned+=("$arg")
                ;;
        esac
    done

    if [[ ${#cleaned[@]} -gt 0 ]]; then
        printf '%s\n' "${cleaned[@]}"
    fi
}

clean_no_ray_command() {
    local args=("$@")
    local cleaned=()
    local i arg
    for ((i = 0; i < ${#args[@]}; i++)); do
        arg="${args[$i]}"
        case "$arg" in
            --distributed-executor-backend)
                ((i++))
                ;;
            --distributed-executor-backend=*)
                ;;
            *)
                cleaned+=("$arg")
                ;;
        esac
    done

    if [[ ${#cleaned[@]} -gt 0 ]]; then
        printf '%s\n' "${cleaned[@]}"
    fi
}

parse_worker_nodes() {
    WORKER_NODES=()
    [[ -z "$WORKERS_CSV" ]] && return 0

    local raw_nodes=()
    local worker
    IFS=',' read -r -a raw_nodes <<< "$WORKERS_CSV"

    for worker in "${raw_nodes[@]}"; do
        worker="$(echo "$worker" | xargs)"
        [[ -n "$worker" ]] && WORKER_NODES+=("$worker")
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-ray)
            MODE="no-ray"
            shift
            ;;
        --workers)
            if [[ $# -lt 2 ]]; then
                echo "Error: --workers requires a comma-separated IP list."
                exit 1
            fi
            WORKERS_CSV="$2"
            shift 2
            ;;
        --master-port|--ray-port)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a port number."
                exit 1
            fi
            MASTER_PORT="$2"
            shift 2
            ;;
        --container-name)
            if [[ $# -lt 2 ]]; then
                echo "Error: --container-name requires a value."
                exit 1
            fi
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            CLUSTER_COMMAND=("$@")
            break
            ;;
        *)
            DOCKER_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
    echo "Error: Invalid port: $MASTER_PORT"
    exit 1
fi

if VLLM_HOST_IP=$(extract_vllm_host_ip "${DOCKER_ARGS[@]}"); then
    :
else
    VLLM_HOST_IP=""
fi

if [[ "$NODE_TYPE" == "--head" && -n "$VLLM_HOST_IP" && "$VLLM_HOST_IP" != "$HEAD_NODE_ADDRESS" ]]; then
    echo "Warning: VLLM_HOST_IP ($VLLM_HOST_IP) differs from head_node_ip ($HEAD_NODE_ADDRESS)."
    echo "Using VLLM_HOST_IP as the head node address."
    HEAD_NODE_ADDRESS="$VLLM_HOST_IP"
fi

mapfile -t BASE_DOCKER_ARGS < <(strip_vllm_host_ip_env "${DOCKER_ARGS[@]}")
parse_worker_nodes

build_node_docker_args() {
    local node_ip="$1"
    local out=("${BASE_DOCKER_ARGS[@]}" "-e" "VLLM_HOST_IP=$node_ip")
    printf '%s\n' "${out[@]}"
}

check_ssh_or_die() {
    local worker_ip="$1"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
        "$worker_ip" true 2>/dev/null; then
        echo "Error: Passwordless SSH to $worker_ip failed."
        exit 1
    fi
}

stop_head_driven_containers() {
    local worker

    echo "Stopping head container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

    for worker in "${WORKER_NODES[@]}"; do
        echo "Stopping worker container on $worker: $CONTAINER_NAME"
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker" \
            "docker rm -f $(printf '%q' "$CONTAINER_NAME") >/dev/null 2>&1 || true" \
            >/dev/null 2>&1 || true
    done
}

cleanup() {
    set +e
    if [[ "$HEAD_DRIVEN_ACTIVE" == "true" ]]; then
        stop_head_driven_containers
    else
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

start_idle_container_local() {
    local node_ip="$1"
    local node_args=()
    mapfile -t node_args < <(build_node_docker_args "$node_ip")

    docker run \
        --entrypoint /bin/bash \
        --network host \
        --name "$CONTAINER_NAME" \
        --gpus all \
        -d \
        --rm \
        -v "$PATH_TO_HF_HOME:/root/.cache/huggingface" \
        "${node_args[@]}" \
        "$DOCKER_IMAGE" -lc 'sleep infinity' >/dev/null
}

start_idle_container_remote() {
    local worker_ip="$1"
    local node_args=()
    local remote_args=()
    local remote_cmd

    mapfile -t node_args < <(build_node_docker_args "$worker_ip")

    remote_args=(
        docker run
        --entrypoint /bin/bash
        --network host
        --name "$CONTAINER_NAME"
        --gpus all
        -d
        --rm
        -v "$PATH_TO_HF_HOME:/root/.cache/huggingface"
        "${node_args[@]}"
        "$DOCKER_IMAGE"
        -lc
        "sleep infinity"
    )

    remote_cmd=$(shell_join "${remote_args[@]}")
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker_ip" "$remote_cmd" >/dev/null
}

validate_head_driven_request() {
    local mode_name="$1"
    local worker

    if [[ "$NODE_TYPE" != "--head" ]]; then
        echo "Error: $mode_name is head-driven. Run it only on the head node with --head."
        exit 1
    fi
    if [[ ${#WORKER_NODES[@]} -eq 0 ]]; then
        echo "Error: $mode_name requires --workers <ip1,ip2,...>."
        exit 1
    fi
    if [[ ${#CLUSTER_COMMAND[@]} -eq 0 ]]; then
        echo "Error: $mode_name requires a command after '--'."
        exit 1
    fi

    for worker in "${WORKER_NODES[@]}"; do
        check_ssh_or_die "$worker"
    done
}

start_all_idle_containers() {
    local worker

    HEAD_DRIVEN_ACTIVE="true"

    echo "Starting idle head container: $CONTAINER_NAME"
    start_idle_container_local "$HEAD_NODE_ADDRESS"

    for worker in "${WORKER_NODES[@]}"; do
        echo "Starting idle worker container on $worker: $CONTAINER_NAME"
        start_idle_container_remote "$worker"
    done
}

start_ray_head() {
    local ray_cmd=(
        ray start
        --block
        --head
        "--node-ip-address=$HEAD_NODE_ADDRESS"
        "--port=$MASTER_PORT"
    )
    local ray_cmd_str
    ray_cmd_str=$(shell_join "${ray_cmd[@]}")

    echo "Starting Ray head on $HEAD_NODE_ADDRESS:$MASTER_PORT"
    docker exec -d "$CONTAINER_NAME" bash -lc \
        "$ray_cmd_str >> /proc/1/fd/1 2>&1"
}

wait_for_ray_head() {
    local attempt
    for ((attempt = 1; attempt <= 60; attempt++)); do
        if docker exec "$CONTAINER_NAME" ray status >/dev/null 2>&1; then
            echo "Ray head is ready."
            return 0
        fi
        sleep 1
    done

    echo "Error: Ray head did not become ready."
    echo "Head container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 || true
    exit 1
}

start_ray_worker() {
    local worker_ip="$1"
    local ray_cmd=(
        ray start
        --block
        "--address=$HEAD_NODE_ADDRESS:$MASTER_PORT"
        "--node-ip-address=$worker_ip"
    )
    local ray_cmd_str payload remote_cmd

    ray_cmd_str=$(shell_join "${ray_cmd[@]}")
    payload="$ray_cmd_str >> /proc/1/fd/1 2>&1"
    printf -v remote_cmd 'docker exec -d %q bash -lc %q' \
        "$CONTAINER_NAME" "$payload"

    echo "Starting Ray worker on $worker_ip"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker_ip" "$remote_cmd"
}

wait_for_ray_cluster() {
    local expected_nodes="$1"
    local attempt
    local check_code

    printf -v check_code '%s' \
'import ray, sys
try:
    ray.init(address="'"$HEAD_NODE_ADDRESS:$MASTER_PORT"'", logging_level="ERROR")
    alive = sum(1 for node in ray.nodes() if node.get("Alive"))
    ray.shutdown()
    print(alive)
    sys.exit(0 if alive >= '"$expected_nodes"' else 1)
except Exception:
    sys.exit(1)'

    for ((attempt = 1; attempt <= 90; attempt++)); do
        if docker exec "$CONTAINER_NAME" python3 -c "$check_code" >/dev/null 2>&1; then
            echo "Ray cluster is ready: $expected_nodes node(s)."
            docker exec "$CONTAINER_NAME" ray status || true
            return 0
        fi
        sleep 1
    done

    echo "Error: Ray cluster did not reach $expected_nodes active node(s)."
    echo "Ray status:"
    docker exec "$CONTAINER_NAME" ray status 2>&1 || true
    echo "Head container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 || true
    exit 1
}

execute_on_head() {
    local command_str
    command_str=$(shell_join "${CLUSTER_COMMAND[@]}")

    echo "Launching command on Ray head container:"
    printf '  %s\n' "$command_str"

    if [[ -t 0 ]]; then
        docker exec -it "$CONTAINER_NAME" bash -lc "$command_str"
    else
        docker exec -i "$CONTAINER_NAME" bash -lc "$command_str"
    fi
}

run_managed_ray_mode() {
    local worker
    local expected_nodes

    validate_head_driven_request "Managed Ray mode"
    start_all_idle_containers

    start_ray_head
    wait_for_ray_head

    for worker in "${WORKER_NODES[@]}"; do
        start_ray_worker "$worker"
    done

    expected_nodes=$((1 + ${#WORKER_NODES[@]}))
    wait_for_ray_cluster "$expected_nodes"
    execute_on_head
}

run_no_ray_mode() {
    local clean_cmd=()
    local total_nodes
    local rank=1
    local worker
    local worker_cmd=()
    local worker_cmd_str
    local payload remote_cmd
    local head_cmd=()
    local head_cmd_str

    validate_head_driven_request "No-Ray mode"
    mapfile -t clean_cmd < <(clean_no_ray_command "${CLUSTER_COMMAND[@]}")

    total_nodes=$((1 + ${#WORKER_NODES[@]}))
    start_all_idle_containers

    for worker in "${WORKER_NODES[@]}"; do
        worker_cmd=(
            "${clean_cmd[@]}"
            --nnodes "$total_nodes"
            --node-rank "$rank"
            --master-addr "$HEAD_NODE_ADDRESS"
            --master-port "$MASTER_PORT"
            --headless
        )
        worker_cmd_str=$(shell_join "${worker_cmd[@]}")
        payload="$worker_cmd_str >> /proc/1/fd/1 2>&1"
        printf -v remote_cmd 'docker exec -d %q bash -lc %q' \
            "$CONTAINER_NAME" "$payload"

        echo "Launching no-ray worker rank $rank on $worker"
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker" "$remote_cmd"
        ((rank++))
    done

    head_cmd=(
        "${clean_cmd[@]}"
        --nnodes "$total_nodes"
        --node-rank 0
        --master-addr "$HEAD_NODE_ADDRESS"
        --master-port "$MASTER_PORT"
    )
    head_cmd_str=$(shell_join "${head_cmd[@]}")

    echo "Launching no-ray head rank 0 on $HEAD_NODE_ADDRESS"
    if [[ -t 0 ]]; then
        docker exec -it "$CONTAINER_NAME" bash -lc "$head_cmd_str"
    else
        docker exec -i "$CONTAINER_NAME" bash -lc "$head_cmd_str"
    fi
}

run_legacy_ray_mode() {
    local legacy_ray_port="6379"
    local ray_start_cmd="ray start --block"
    local run_args=()

    if [[ "$NODE_TYPE" == "--head" ]]; then
        ray_start_cmd+=" --head --node-ip-address=${HEAD_NODE_ADDRESS} --port=${legacy_ray_port}"
    else
        ray_start_cmd+=" --address=${HEAD_NODE_ADDRESS}:${legacy_ray_port}"
        if [[ -n "$VLLM_HOST_IP" ]]; then
            ray_start_cmd+=" --node-ip-address=${VLLM_HOST_IP}"
        fi
    fi

    run_args=(
        --entrypoint /bin/bash
        --network host
        --name "$CONTAINER_NAME"
        --gpus all
        -v "$PATH_TO_HF_HOME:/root/.cache/huggingface"
        "${BASE_DOCKER_ARGS[@]}"
        "$DOCKER_IMAGE"
        -lc
        "$ray_start_cmd"
    )

    docker run "${run_args[@]}"
}

if [[ "$MODE" == "no-ray" ]]; then
    run_no_ray_mode
elif [[ -n "$WORKERS_CSV" || ${#CLUSTER_COMMAND[@]} -gt 0 ]]; then
    run_managed_ray_mode
else
    run_legacy_ray_mode
fi
