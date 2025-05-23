# Validating in the SoundsRight Subnet 

## Summary
Running a validator the in Subnet requires **1,000 staked TAO**. 

**We also implore validators to run:**
1. **In a separate environment dedicated to validating for only the SoundsRight subnet.**
2. **Using a child hotkey.**

## Validator deployment 

### 1. Virtual machine deployment
The subnet requires **Ubuntu 24.04** and **Python 3.12** with at least the following hardware configuration:

- 16 GB VRAM
- 23 GB RAM
- 512 GB storage (1000 IOPS)
- 5 gbit/s network bandwidth
- 6 vCPU 

When running the subnet validator, we are highly recommending that you run the subnet validator with DataCrunch.io using the **1x Tesla V100** instance type with **Ubuntu 24.04** and **CUDA 12.6**. 

This is the setup we are performing our testing and development with; as a result, they are being used as the performance baseline for the subnet validators.

Running the validator with DataCrunch.io is not mandatory and the subnet validator should work on other environments as well, though the exact steps for setup may vary depending on the service used. This guide assumes you're running Ubuntu 24.04 provided by DataCrunch.io, and thus skips steps that might be mandatory in other environments (for example, installing the NVIDIA and CUDA drivers).

### 2. Installation of mandatory packages

Note that for the following steps, it will be assumed that you will be running the validator fully as root and as such, any action that needs to be performed as root will not be denoted with sudo.

#### 2.1 Install Podman for Ubuntu 

For installing Podman for Ubuntu, run the following command:
```
$ apt-get update
$ apt-get -y install podman
```

#### 2.2 Install the mandatory packages

Run the following command:
```
$ apt update && apt-get install python3.12-venv && apt install jq && apt install npm && npm install pm2 -g && pm2 update && apt install -y python3.12-dev build-essential gcc g++
```

#### 2.3 Configure NVIDIA Container Toolkit and CDI

Follow the instructions to download the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) with Apt.

Modify `/etc/nvidia-container-runtime/config.toml` and set the following parameters if you're running docker as non-root user:
```
[nvidia-container-cli]
no-cgroups = true

[nvidia-container-runtime]
debug = "/tmp/nvidia-container-runtime.log"
```
You can also run the following command to achieve the same result:
```
$ sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
```

Next, follow the instructions for [generating a CDI specification](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

Verify that the CDI specification was done correctly with:
```
$ nvidia-ctk cdi list
```
You should see this in your output:
```
nvidia.com/gpu=all
nvidia.com/gpu=0
```

### 3. Preparation

This section covers setting up the repository, virtual environment, regenerating wallets, and setting up environmental variables.

#### 3.1 Setup the GitHub repository and python virtualenv
To clone the repository and setup the Python virtualenv, execute the following commands:
```
$ git clone https://github.com/synapsec-ai/soundsright-subnet.git
$ cd soundsright-subnet
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install bittensor
```

#### 3.2 Regenerate the validator wallet

The private portion of the coldkey is not needed to run the subnet validator. **Never have your private validator coldkey or hotkeys not used to run the validator stored on the server**. Please use a dedicated server for each subnet to minimize impact of potential security issues.

To regenerate the keys on the host, execute the following commands:
```
(.venv) $ btcli wallet regen_coldkeypub
(.venv) $ btcli wallet regen_hotkey
```

#### 3.3 Setup the environmental variables
The subnet repository contains a sample validator env (`.env.sample`) file that is used to pass the correct parameters to the docker compose file.

Create a new file in the root of the repository called `.env` based on the given sample.
```
(.venv) $ cp .validator-env.sample .env
```
The contents of the `.env` file must be adjusted according to the validator configuration. Below is a table explaining what each variable in the .env file represents (note that the .env variables that do not apply for validators are not listed here):

| Variable | Meaning |
| :------: | :-----: |
| NETUID | The subnet's netuid. For mainnet this value is , and for testnet this value is 271. |
| SUBTENSOR_CHAIN_ENDPOINT | The Bittensor chain endpoint. Please make sure to always use your own endpoint. For mainnnet, the default endpoint is: wss://finney.opentensor.ai:443, and for testnet the default endpoint is: wss://test.finney.opentensor.ai:443. |
| WALLET | The name of your coldkey. |
| HOTKEY | The name of your hotkey. |
| LOG_LEVEL | Specifies the level of logging you will see on the validator. Choose between INFO, INFOX, DEBUG. DEBUGX, TRACE, and TRACEX. |
| OPENAI_API_KEY | Your OpenAI API key. |
| HEALTHCHECK_API_HOST | Host for HealthCheck API, default is 0.0.0.0. There is no need to adjust this value unless you want to. |
| HEALTHCHECK_API_PORT | Port for HealthCheck API, default is 6000. There is no need to adjust this value unless you want to. |

.env example:
```
NETUID=
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET=my_coldkey
HOTKEY=my_hotkey
OPENAI_API_KEY=THIS-IS-AN-OPENAI-API-KEY-wfhwe78r78frfg7e8ghrveh78ehrg

# Available: INFO, INFOX, DEBUG, DEBUGX, TRACE, TRACEX
LOG_LEVEL=TRACE

# HealthCheck API
HEALTHCHECK_API_HOST=0.0.0.0
HEALTHCHECK_API_PORT=6000
```

#### 3.4 Installing Python Dependencies

Run the following commands:

```
(.venv) $ pip install --use-pep517 pesq==0.0.4 && pip install -e .[validator] && pip install httpx==0.27.2
```

### 4. Running the validator

Run the validator with this command: 
```
$ bash scripts/run_validator.sh --name soundsright-validator --max_memory_restart 50G --branch main
```
To see the logs, execute the following command: 
```
$ pm2 logs <process-name-or-id>
``` 

### 5. Updating validator

To update the validator, pull the newest changes to main and restart the pm2 process:

```
$ cd soundsright-subnet
$ git pull && pm2 restart
```

### 6. Assessing validator health 

A HealthCheck API is built into the validator, which can be queried for an assessment of the validator's performance. Note that the commands in this section assume default values for the `healthcheck_host` and `healthcheck_port` arguments of `0.0.0.0` and `6000` respectively. The following endpoints are available: 

#### 6.1 Healthcheck

This endpoint offers an overview of validator performance. It can be queried with:

```
$ curl http://127.0.0.1:6000/healthcheck | jq
```

#### 6.2 Metrics 

This endpoint offers a view of all of the metrics tabulated by the Healthcheck API. It can be queried with:
```
$ curl http://127.0.0.1:6000/healthcheck/metrics | jq
```

#### 6.3 Events 

This endpoint offers insight into WARNING, SUCCESS and ERROR logs in the validator. It can be queried with:
```
$ curl http://127.0.0.1:6000/healthcheck/events | jq
```

#### 6.4 Best Models by Competition

This endpoint offers insight into the best models known by the validator for the previous day's competition. It can be queried with:
```
$ curl http://127.0.0.1:6000/healthcheck/best_models | jq
```

#### 6.5 Models for Current Competitions
This endpoint offers insight into the best models known by the validator for the previous day's competition. It can be queried with:
```
$ curl http://127.0.0.1:6000/healthcheck/current_models | jq
```