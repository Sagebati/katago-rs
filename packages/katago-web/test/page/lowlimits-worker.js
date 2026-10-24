// Test double: the real worker on a device with WebGPU's *default* limits (maxStorageBufferBindingSize 128 MiB,
// maxBufferSize 256 MiB), as on adapters that expose no more: the engine requests the adapter's limits, which on
// this machine's Vulkan adapter are 4 GiB, so the cap is applied to the device request. Used for b18c384nbt, whose
// biggest buffer, the im2col matrix of its 3x3 convs, is 2.5 MB per batch row: 80 MB at 32 rows (fits), 160 MB at 64
// (does not fit a default-limit binding: the NN backend fails with "readback failed").
const CAP = { maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456 };
const proto = globalThis.GPUAdapter?.prototype;
if (proto) {
  const requestDevice = proto.requestDevice;
  proto.requestDevice = function (desc = {}) {
    const requiredLimits = { ...(desc.requiredLimits ?? {}) };
    for (const [k, v] of Object.entries(CAP)) requiredLimits[k] = Math.min(requiredLimits[k] ?? v, v);
    return requestDevice.call(this, { ...desc, requiredLimits }).then((device) => {
      console.log(`lowlimits: device maxStorageBufferBindingSize ${device.limits.maxStorageBufferBindingSize}, maxBufferSize ${device.limits.maxBufferSize}`);
      return device;
    });
  };
}
// Messages that arrive while worker.js is still loading are replayed to it.
const early = [];
const hold = (e) => early.push(e.data);
self.addEventListener("message", hold);
await import("/pkg/worker.js");
self.removeEventListener("message", hold);
for (const data of early) self.dispatchEvent(new MessageEvent("message", { data }));
