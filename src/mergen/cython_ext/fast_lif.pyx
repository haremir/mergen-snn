# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp

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
