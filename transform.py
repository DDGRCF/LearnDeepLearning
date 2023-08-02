import numpy as np
from copy import deepcopy

def softmax(x: np.ndarray, dim : int) -> np.ndarray:
    x = np.exp(x)
    s = np.sum(x, axis = dim, keepdims = True) # dim = 0 [1, 4]
    return x / s


if __name__ == "__main__":
    n = 4
    dim1 = 1024
    dim2 = 256
    x = np.random.randint(0, 2, (n, dim1))
    x = x.astype(np.float32)
    dim_t = np.arange(dim1, dtype=np.float32)
    dim_t = 10000 ** (dim_t // 2 / dim1)
    print(dim_t.shape)

    w1 = np.random.rand(dim1)
    x_embed = np.arange(dim1, dtype=np.float32) * w1

    pos = x_embed / dim_t
    pos[0::2] = np.sin(pos[0::2])
    pos[1::2] = np.cos(pos[1::2])
    
    x += pos[None]
        
    w2 = np.random.rand(dim2, dim1) 

    a = x @ w2.T;
    print(a.shape)

    wq = np.random.rand(512, dim2)
    wk = deepcopy(wq)
    wv = deepcopy(wq)

    q = a @ wq.T
    k = a @ wk.T
    v = a @ wv.T

    q = q / np.max(q)
    k = k / np.max(k)
    v = v / np.max(v)

    print(q.shape)
    print(k.shape)
    print(v.shape)

    s = q @ k.T * (q.shape[-1]) ** -0.5
    print(s.shape)
    s = softmax(s, dim = 1)
    print(s.shape)

    s = s.reshape(n, 1, n)
    vs = s @ v
    vs = vs.reshape(n, -1)
    print(vs.shape)
