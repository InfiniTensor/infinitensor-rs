#![allow(unused)]

use common::DataType;
use std::{collections::HashMap, time::Duration};

#[repr(transparent)]
pub struct KernelLib(HashMap<String, DeviceLib>);

#[repr(transparent)]
struct DeviceLib(HashMap<String, Kernel>);

pub struct Kernel {
    code: *const (),
    perf_data: Vec<PerfData>,
}

pub struct PerfData {
    params: Vec<PerfParam>,
    time: Duration,
    memory: usize,
}

pub struct PerfParam {
    dtype: DataType,
    shape: Vec<usize>,
}

impl KernelLib {
    pub fn push_kernel(&mut self, device: String, name: String, code: *const ()) {
        self.0
            .entry(device)
            .or_insert(DeviceLib(HashMap::new()))
            .0
            .insert(
                name,
                Kernel {
                    code,
                    perf_data: Vec::new(),
                },
            );
    }

    pub fn push_record(
        &mut self,
        device: String,
        kernel: String,
        params: Vec<PerfParam>,
        time: Duration,
        memory: usize,
    ) {
        self.0
            .get_mut(&device)
            .unwrap()
            .0
            .get_mut(&kernel)
            .unwrap()
            .perf_data
            .push(PerfData {
                params,
                time,
                memory,
            });
    }
}
