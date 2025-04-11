# simple_test.py
import jax, jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import traceback
import os # Import os for environment check, if needed

err = None
res = None
print("Starting simple_test.py...")

try:
    # Optional: Print env vars for debugging TPU environment
    # print(f"Env Vars: JAX_PROCESS_COUNT={os.environ.get('JAX_PROCESS_COUNT')}, JAX_PROCESS_INDEX={os.environ.get('JAX_PROCESS_INDEX')}")

    print("Initializing JAX distributed...")
    jax.distributed.initialize()
    print("JAX distributed initialized.")

    pc = jax.process_count()
    pid = jax.process_index()
    ldc = jax.local_device_count()
    gdc = jax.device_count()
    print(f"Process {pid}/{pc}, Local Devices={ldc}, Global Devices={gdc}")

    mesh = None
    if pc > 1:
        print(f"P{pid}: Attempting mesh creation ({pc} processes, {ldc} local devices)...")
        try:
            # Create the mesh
            mesh = Mesh(mesh_utils.create_device_mesh((pc, ldc)), ("i","d")) # Use double quotes for axis names
            print(f"P{pid}: Mesh created OK. Axis names: {mesh.axis_names}, Device shape: {mesh.devices.shape}")

            # Attempt pmean
            print(f"P{pid}: Attempting pmean with axis_name='i'...")
            local_value = jnp.array(float(pid)) # Value specific to this process
            with mesh:
                res = jax.lax.pmean(local_value, axis_name="i") # Use double quotes

            print(f"P{pid}: Pmean successful! Result = {res}")

            # Optional check of the result
            expected = (pc - 1) / 2.0
            if jnp.isclose(res, expected):
                print(f"P{pid}: Result matches expected value {expected:.2f}")
            else:
                print(f"P{pid}: WARNING: Result {res:.2f} does NOT match expected {expected:.2f}")

        except Exception as e:
            err = e
            print(f"P{pid}: !!! FAIL during mesh creation or pmean: {err}")
            print("--- Traceback ---")
            traceback.print_exc()
            print("-----------------")
    else:
        print(f"P{pid}: Skipping mesh/pmean test (process_count <= 1)")

except Exception as e:
     err = e
     print(f"P{pid}: !!! FAIL during JAX initialization or setup: {err}")
     print("--- Traceback ---")
     traceback.print_exc()
     print("-----------------")

print(f"P{pid}: simple_test.py Finished.")