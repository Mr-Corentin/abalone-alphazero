# simple_test_pmap.py
import jax, jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import traceback

print("Starting simple_test_pmap.py...")

try:
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
            mesh = Mesh(mesh_utils.create_device_mesh((pc, ldc)), ("i","d")) # Axes: processes, local_devices
            print(f"P{pid}: Mesh created OK. Axis names: {mesh.axis_names}, Device shape: {mesh.devices.shape}")

            # Define the function to be pmapped across the process axis 'i'
            # Note: We operate on the 'i' axis of the mesh.
            # The input 'process_value' will be specific to each process within the pmap context.
            def mean_across_processes(process_value):
                # pmean now operates on the axis named 'i' which is being mapped by pmap
                return jax.lax.pmean(process_value, axis_name='i')

            # Create a value specific to this process (e.g., its ID)
            # We need to ensure this value exists *before* calling pmap.
            # For pmap over the process axis, the input shape usually doesn't need
            # modification if we want each process to contribute its scalar value.
            my_value = jnp.array(float(pid))
            print(f"P{pid}: Value for this process = {my_value}")

            # Apply pmap over the 'i' axis (processes).
            # We target devices based on the mesh definition for the 'i' axis.
            # Use mesh.devices[:, 0] to get one device per process row in the mesh.
            # Note: JAX often figures this out with just the mesh context, but being explicit can help.
            print(f"P{pid}: Attempting pmap over axis 'i'...")

            # Method 1: Using the mesh context
            # with mesh: # The context might be enough
            #     pmap_result = jax.pmap(mean_across_processes, axis_name='i')(my_value)

            # Method 2: Explicitly targeting devices for the 'i' axis mapping
            # Get one device per process (first device in each process's row)
            process_devices = mesh.devices[:, 0]
            pmap_result = jax.pmap(mean_across_processes, axis_name='i', devices=process_devices)(my_value)


            # The result of pmap is replicated on the devices used for the map.
            # We expect the mean value.
            print(f"P{pid}: pmap successful!")
            # Result might be slightly different depending on device placement, take first element
            final_mean = pmap_result # pmap might return it on the first device mapped. Adjust if needed.
            print(f"P{pid}: Mean calculated via pmap = {final_mean}")

            # Check the result
            expected = (pc - 1) / 2.0
            if jnp.isclose(final_mean, expected):
                print(f"P{pid}: Result matches expected value {expected:.2f}")
            else:
                print(f"P{pid}: WARNING: Result {final_mean:.2f} does NOT match expected {expected:.2f}")

        except Exception as e:
            print(f"P{pid}: !!! FAIL during mesh creation or pmap/pmean: {e}")
            print("--- Traceback ---")
            traceback.print_exc()
            print("-----------------")
    else:
        print(f"P{pid}: Skipping mesh/pmap test (process_count <= 1)")

except Exception as e:
     print(f"P{pid}: !!! FAIL during JAX initialization or setup: {err}")
     print("--- Traceback ---")
     traceback.print_exc()
     print("-----------------")

print(f"P{pid}: simple_test_pmap.py Finished.")