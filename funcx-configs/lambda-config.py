from funcx_endpoint.endpoint.utils.config import Config
from funcx_endpoint.executors import HighThroughputExecutor
from funcx_endpoint.strategies.simple import SimpleStrategy
from parsl.providers import LocalProvider

config = Config(
    executors=[HighThroughputExecutor(
        strategy=SimpleStrategy(max_idletime=6000),
        heartbeat_threshold=7200,
        available_accelerators=20,
        provider=LocalProvider(
            init_blocks=1,
            min_blocks=0,
            max_blocks=1,
            worker_init='''export TF_FORCE_GPU_ALLOW_GROWTH=true'''
        ),
    )],
    funcx_service_address='https://api2.funcx.org/v2'
)

# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": "default",
    "description": "",
    "organization": "",
    "department": "",
    "public": False,
    "visible_to": []
}
