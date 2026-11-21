//! The runtimes' device listings (`cuda::devices`, `hip::devices`,
//! `metal::devices`): a machine's devices by index, without opening one.
#![cfg(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))]

use arael_gpu::{Device, DeviceInfo};

/// `listed` has device `index` as the opened device `dev` describes itself.
fn listed_as_opened<D: Device>(listed: &[DeviceInfo], index: usize, dev: &D) {
    let info = listed.iter().find(|d| d.index == index).unwrap_or_else(|| panic!("device {index} in {listed:?}"));
    let arch = dev.arch();
    assert_eq!((info.name.as_str(), info.integrated), (arch.name.as_str(), arch.integrated), "{listed:?}");
}

/// CUDA lists the devices it opens, by index, each with the name and the
/// integrated flag the opened device reports, and opens none past its list.
#[cfg(feature = "cuda")]
#[test]
fn cuda_lists_the_devices_it_opens() {
    use arael_gpu::cuda::{CudaDevice, devices};
    let listed = devices().unwrap();
    assert!(!listed.is_empty() && listed.iter().enumerate().all(|(i, d)| d.index == i), "{listed:?}");
    for d in &listed {
        listed_as_opened(&listed, d.index, &CudaDevice::new(d.index).unwrap());
    }
    assert!(matches!(CudaDevice::new(listed.len()), Err(arael_gpu::GpuError::Unavailable(_))));
}

/// HIP lists the devices it opens, as CUDA does.
#[cfg(feature = "hip")]
#[test]
fn hip_lists_the_devices_it_opens() {
    use arael_gpu::hip::{HipDevice, devices};
    let listed = devices().unwrap();
    assert!(!listed.is_empty() && listed.iter().enumerate().all(|(i, d)| d.index == i), "{listed:?}");
    for d in &listed {
        listed_as_opened(&listed, d.index, &HipDevice::new(d.index).unwrap());
    }
    assert!(matches!(HipDevice::new(listed.len()), Err(arael_gpu::GpuError::Unavailable(_))));
}

/// Metal lists the Apple GPUs it opens, each with its index among the
/// Mac's Metal devices.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_lists_the_devices_it_opens() {
    use arael_gpu::metal::{MetalDevice, devices};
    let listed = devices();
    assert!(!listed.is_empty(), "no Apple GPU");
    for d in &listed {
        listed_as_opened(&listed, d.index, &MetalDevice::new(d.index).unwrap());
    }
}
