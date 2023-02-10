from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
from parsl.addresses import address_by_interface
from funcx_endpoint.endpoint.utils.config import Config
from funcx_endpoint.executors import HighThroughputExecutor

config = Config(
    executors=[HighThroughputExecutor(
        max_workers_per_node=1,
        address=address_by_interface('vlan2360'),
        provider=CobaltProvider(
            queue='debug-flat-quad',
            account='CSC249ADCD08',
            launcher=AprunLauncher(overrides="-d 256 --cc depth -j 4"),
            worker_init='''
module load miniconda-3
source activate /lus/theta-fs0/projects/CSC249ADCD08/multi-site-campaigns/env
which python
	    ''',
            nodes_per_block=8,
            init_blocks=0,
            min_blocks=0,
            max_blocks=1,
            cmd_timeout=300,
            walltime='00:60:00',
            scheduler_options='#COBALT --attrs enable_ssh=1:filesystems=home,theta-fs0'
        ),
    )],
    funcx_service_address='https://api2.funcx.org/v2'
)

# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": "edw-debug-single-node",
    "description": "Endpoint to run single-node workloads for the electrolyte design workflows on debug queue",
    "organization": "Argonne National Laboratory",
    "department": "Data Science and Learning Division",
    "public": False,
    "visible_to": []
}
