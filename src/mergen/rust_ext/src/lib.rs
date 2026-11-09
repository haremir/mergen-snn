use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use rayon::prelude::*;

#[pyfunction]
#[pyo3(signature = (x, mem_init, beta, threshold, return_all_mem=true))]
#[allow(unused_variables)]
fn fast_lif_sequence_forward<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArray3<f32>>,
    mem_init: Bound<'py, PyArray2<f32>>,
    beta: f32,
    threshold: f32,
    return_all_mem: bool,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray3<f32>>)> {
    let shape_x = x.shape();
    let b_size = shape_x[0];
    let t_steps = shape_x[1];
    let h_size = shape_x[2];

    let shape_mem = mem_init.shape();
    if shape_mem[0] != b_size || shape_mem[1] != h_size {
        return Err(PyValueError::new_err(format!(
            "mem_init shape {:?} does not match expected ({}, {})",
            shape_mem, b_size, h_size
        )));
    }

    let x_ro = x.readonly();
    let x_slice = x_ro.as_slice()?;
    let mem_init_ro = mem_init.readonly();
    let mem_init_slice = mem_init_ro.as_slice()?;

    let spk_arr = PyArray3::zeros_bound(py, [b_size, t_steps, h_size], false);
    let mem_all_arr = PyArray3::zeros_bound(py, [b_size, t_steps, h_size], false);
    let mem_out_arr = PyArray2::zeros_bound(py, [b_size, h_size], false);

    let spk_slice = unsafe { spk_arr.as_slice_mut()? };
    let mem_all_slice = unsafe { mem_all_arr.as_slice_mut()? };
    let mem_out_slice = unsafe { mem_out_arr.as_slice_mut()? };

    let th = threshold;
    let half_th = 0.5f32 * threshold;

    // Parallelize over batches using Rayon without Python GIL
    py.allow_threads(|| {
        let chunk_th = t_steps * h_size;

        spk_slice
            .par_chunks_exact_mut(chunk_th)
            .zip(mem_all_slice.par_chunks_exact_mut(chunk_th))
            .zip(mem_out_slice.par_chunks_exact_mut(h_size))
            .zip(x_slice.par_chunks_exact(chunk_th))
            .zip(mem_init_slice.par_chunks_exact(h_size))
            .for_each(|((((spk_b, mem_all_b), mem_out_b), x_b), mem_init_b)| {
                for h in 0..h_size {
                    let mut curr_m = mem_init_b[h];
                    for t in 0..t_steps {
                        let idx = t * h_size + h;
                        let m = curr_m * beta + x_b[idx];
                        mem_all_b[idx] = m;
                        if m > th {
                            spk_b[idx] = 1.0;
                            curr_m = m - half_th;
                        } else {
                            spk_b[idx] = 0.0;
                            curr_m = m;
                        }
                        if curr_m < -0.5 {
                            curr_m = -0.5;
                        }
                    }
                    mem_out_b[h] = curr_m;
                }
            });
    });

    Ok((spk_arr, mem_out_arr, mem_all_arr))
}

#[pyfunction]
#[pyo3(signature = (grad_spk, spk, mem, mem_init, beta, threshold, alpha=2.0, grad_mem_out=None))]
#[allow(unused_variables)]
fn fast_lif_sequence_backward<'py>(
    py: Python<'py>,
    grad_spk: Bound<'py, PyArray3<f32>>,
    spk: Bound<'py, PyArray3<f32>>,
    mem: Bound<'py, PyArray3<f32>>,
    mem_init: Bound<'py, PyArray2<f32>>,
    beta: f32,
    threshold: f32,
    alpha: f32,
    grad_mem_out: Option<Bound<'py, PyArray2<f32>>>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>)> {
    let shape = grad_spk.shape();
    let b_size = shape[0];
    let t_steps = shape[1];
    let h_size = shape[2];

    let grad_spk_ro = grad_spk.readonly();
    let grad_spk_slice = grad_spk_ro.as_slice()?;

    let mem_ro = mem.readonly();
    let mem_slice = mem_ro.as_slice()?;

    let grad_mem_out_vec: Option<Vec<f32>> = match grad_mem_out {
        Some(arr) => Some(arr.readonly().as_slice()?.to_vec()),
        None => None,
    };

    let grad_x_arr = PyArray3::zeros_bound(py, [b_size, t_steps, h_size], false);
    let grad_mem_init_arr = PyArray2::zeros_bound(py, [b_size, h_size], false);

    let grad_x_slice = unsafe { grad_x_arr.as_slice_mut()? };
    let grad_mem_init_slice = unsafe { grad_mem_init_arr.as_slice_mut()? };

    let pi = 3.141592653589793f32;
    let pi_alpha_half = (pi / 2.0) * alpha;

    py.allow_threads(|| {
        let chunk_th = t_steps * h_size;

        grad_x_slice
            .par_chunks_exact_mut(chunk_th)
            .zip(grad_mem_init_slice.par_chunks_exact_mut(h_size))
            .zip(grad_spk_slice.par_chunks_exact(chunk_th))
            .zip(mem_slice.par_chunks_exact(chunk_th))
            .enumerate()
            .for_each(|(b, (((grad_x_b, grad_mem_init_b), grad_spk_b), mem_b))| {
                for h in 0..h_size {
                    let mut rec_grad = if let Some(ref g_mem) = grad_mem_out_vec {
                        g_mem[b * h_size + h]
                    } else {
                        0.0
                    };

                    for t in (0..t_steps).rev() {
                        let idx = t * h_size + h;
                        let shift = mem_b[idx] - threshold;
                        let denom = 1.0 + (pi_alpha_half * shift) * (pi_alpha_half * shift);
                        let surr = (alpha * 0.5) / denom;
                        let grad_m = grad_spk_b[idx] * surr + rec_grad;
                        grad_x_b[idx] = grad_m;
                        rec_grad = grad_m * beta;
                    }
                    grad_mem_init_b[h] = rec_grad;
                }
            });
    });

    Ok((grad_x_arr, grad_mem_init_arr))
}

#[pymodule]
fn mergen_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_lif_sequence_forward, m)?)?;
    m.add_function(wrap_pyfunction!(fast_lif_sequence_backward, m)?)?;
    Ok(())
}
