# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
from cython.parallel import prange

def fast_lif_forward(float[:, :] x, float[:, :] mem, float beta, float threshold,
                     object spk_out=None, object mem_out=None):
    """
    Tek zaman adımı için optimize edilmiş C++ LIF nöron ileri geçişi.
    x: (batch_size, hidden_size) girdi akımı
    mem: (batch_size, hidden_size) mevcut membran potansiyeli
    beta: sönümleme katsayısı (decay rate)
    threshold: ateşleme eşiği
    spk_out, mem_out: isteğe bağlı önceden ayrılmış tensör/dizi tamponları
    """
    cdef int batch_size = x.shape[0]
    cdef int hidden_size = x.shape[1]
    cdef int b, h
    cdef float m

    if spk_out is None:
        spk = np.zeros((batch_size, hidden_size), dtype=np.float32)
    else:
        spk = spk_out
    cdef float[:, :] spk_view = spk

    if mem_out is None:
        new_mem = np.zeros((batch_size, hidden_size), dtype=np.float32)
    else:
        new_mem = mem_out
    cdef float[:, :] new_mem_view = new_mem

    for b in range(batch_size):
        for h in range(hidden_size):
            m = mem[b, h] * beta + x[b, h]
            if m > threshold:
                spk_view[b, h] = 1.0
                new_mem_view[b, h] = m - threshold  # Reset by subtraction
            else:
                spk_view[b, h] = 0.0
                new_mem_view[b, h] = m

    return spk, new_mem


def fast_lif_sequence_forward(float[:, :, :] x, float[:, :] mem_init, float beta, float threshold,
                              bint return_all_mem=True):
    """
    Tüm zaman adımlarını (num_steps) OpenMP çoklu iş parçacığıyla C++ içinde dönen dizi LIF ileri geçişi.
    x: (batch_size, num_steps, hidden_size)
    mem_init: (batch_size, hidden_size)
    """
    cdef int batch_size = x.shape[0]
    cdef int num_steps = x.shape[1]
    cdef int hidden_size = x.shape[2]
    cdef int b, t, h
    cdef float m, curr_m

    spk_out = np.empty((batch_size, num_steps, hidden_size), dtype=np.float32)
    cdef float[:, :, :] spk_view = spk_out

    mem_all = np.empty((batch_size, num_steps, hidden_size), dtype=np.float32)
    cdef float[:, :, :] mem_all_view = mem_all

    mem_out = np.empty((batch_size, hidden_size), dtype=np.float32)
    cdef float[:, :] mem_out_view = mem_out

    for b in prange(batch_size, nogil=True):
        for h in range(hidden_size):
            curr_m = mem_init[b, h]
            for t in range(num_steps):
                m = curr_m * beta + x[b, t, h]
                mem_all_view[b, t, h] = m
                if m > threshold:
                    spk_view[b, t, h] = 1.0
                    curr_m = m - threshold
                else:
                    spk_view[b, t, h] = 0.0
                    curr_m = m
            mem_out_view[b, h] = curr_m

    if return_all_mem:
        return spk_out, mem_out, mem_all
    return spk_out, mem_out


def fast_lif_sequence_backward(float[:, :, :] grad_spk,
                               float[:, :, :] spk,
                               float[:, :, :] mem,
                               float[:, :] mem_init,
                               float beta,
                               float threshold,
                               float alpha=2.0,
                               object grad_mem_out=None):
    """
    OpenMP ile paralel zaman boyunca geriye yayılan (BPTT) surrogate gradient türev fonksiyonu.
    grad_spk: (batch_size, num_steps, hidden_size)
    mem: (batch_size, num_steps, hidden_size) - ileri geçişteki membran potansiyelleri
    """
    cdef int batch_size = grad_spk.shape[0]
    cdef int num_steps = grad_spk.shape[1]
    cdef int hidden_size = grad_spk.shape[2]
    cdef int b, t, h
    cdef float shift, surr, grad_m, rec_grad
    cdef float pi = 3.141592653589793
    cdef float pi_alpha_half = (pi / 2.0) * alpha

    grad_x = np.empty((batch_size, num_steps, hidden_size), dtype=np.float32)
    cdef float[:, :, :] grad_x_view = grad_x

    grad_mem_init = np.empty((batch_size, hidden_size), dtype=np.float32)
    cdef float[:, :] grad_mem_init_view = grad_mem_init

    cdef float[:, :] grad_mem_out_view = None
    cdef bint has_grad_mem = False
    if grad_mem_out is not None:
        grad_mem_out_view = grad_mem_out
        has_grad_mem = True

    for b in prange(batch_size, nogil=True):
        for h in range(hidden_size):
            if has_grad_mem:
                rec_grad = grad_mem_out_view[b, h]
            else:
                rec_grad = 0.0

            for t in range(num_steps - 1, -1, -1):
                shift = mem[b, t, h] - threshold
                # ATan surrogate türevi: surr = (alpha/2) / (1 + ((pi/2)*alpha*shift)^2)
                surr = (alpha * 0.5) / (1.0 + (pi_alpha_half * shift) * (pi_alpha_half * shift))
                grad_m = grad_spk[b, t, h] * surr + rec_grad
                grad_x_view[b, t, h] = grad_m
                rec_grad = grad_m * beta

            grad_mem_init_view[b, h] = rec_grad

    return grad_x, grad_mem_init
