import torch
import numpy as np
import torch.nn.functional as F
import torch.fft
import torchvision.transforms.functional as TF
VALUE_TYPE = torch.float32
import matplotlib.pyplot as plt

# def compute_entropy(X, Y, eps):
#     """
#     A placeholder function for computing entropy, replace this with your actual entropy calculation.
#     This function computes -p*log(p) to find entropy based on given inputs X and Y.
#     """
#     # Calculate probabilities
#     p = X * Y
#     p = p / torch.sum(p)
#     entropy = -torch.sum(p * torch.log(p + eps))
#     return entropy
import numpy as np
import torch
import torch.nn.functional as F

def compute_entropy(C, N, eps=1e-7):
    """
    Compute entropy using the formula p * log2(p), where p = C / N.
    """
    p = C / N
    return -p * torch.log2(torch.clamp(p, min=eps, max=None))

def to_tensor(A, on_gpu=True):
    if torch.is_tensor(A):
        A_tensor = A.cuda(non_blocking=True) if on_gpu else A
        if A_tensor.ndim == 2:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1]))
        elif A_tensor.ndim == 3:
            A_tensor = torch.reshape(A_tensor, (1, A_tensor.shape[0], A_tensor.shape[1], A_tensor.shape[2]))
        return A_tensor
    else:
        return to_tensor(torch.tensor(A, dtype=torch.float32), on_gpu=on_gpu)

def create_float_tensor(shape, on_gpu, fill_value=None):
    if on_gpu:
        res = torch.empty((shape[0], shape[1], shape[2], shape[3]), device='cuda', dtype=torch.float32)
        if fill_value is not None:
            res.fill_(fill_value)
        return res
    else:
        if fill_value is not None:
            res = np.full((shape[0], shape[1], shape[2], shape[3]), fill_value=fill_value, dtype='float32')
        else:
            res = np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype='float32')
        return torch.tensor(res, dtype=torch.float32)

def corr_target_setup(A):
    return fft(A)

def fft(A):
    return torch.fft.rfft2(A)

def float_compare(A, c):
    return torch.clamp(1 - torch.abs(A - c), 0.0)

def corr_template_setup(B):
    return torch.conj(fft(B))

def ifft(Afft):
    return torch.fft.irfft2(Afft)

def fftconv(A, B):
    return A * B

def corr_apply(A, B, sz, do_rounding=True):
    C = fftconv(A, B)
    C = ifft(C)
    C = C[:sz[0], :sz[1], :sz[2], :sz[3]]
    if do_rounding:
        # C = torch.round(C)
        C = ste_round(C)
    return C

def fft_of_levelsets(A, Q, packing, setup_fn):
    fft_list = []
    for a_start in range(0, Q, packing):
        a_end = min(a_start + packing, Q)
        levelsets = []
        for a in range(a_start, a_end):
            levelsets.append(float_compare(A, a))
        A_cat = torch.cat(levelsets, 0)
        ffts = setup_fn(A_cat)
        fft_list.append((ffts, a_start, a_end))
    return fft_list

class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Perform the forward pass (rounding operation)
        return torch.round(input)
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: gradient during the backward pass is passed unchanged
        return grad_output

ste_round = StraightThroughRound.apply


def mi_loss(A_tensor, B_tensor, M_B, Q_A=8, Q_B=8, overlap=0.5, enable_partial_overlap=False, normalize_mi=True,
            on_gpu=True, save_maps=False):
    eps = 1e-7

    M_A = create_float_tensor(A_tensor.shape, on_gpu, 1.0)

    M_B = M_B.float() #(B_tensor > 0).float() #create_float_tensor(B_tensor.shape, on_gpu, 1.0)

    # Pad for overlap
    if enable_partial_overlap:
        partial_overlap_pad_sz = (round(B.shape[-1] * (1.0 - overlap)), round(B.shape[-2] * (1.0 - overlap)))
        A_tensor = F.pad(A_tensor, (
        partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]),
                         mode='constant', value=Q_A + 1)
        M_A = F.pad(M_A, (
        partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]),
                    mode='constant', value=0)
    else:
        partial_overlap_pad_sz = (0, 0)

    ext_ashape = A_tensor.shape
    ashape = ext_ashape[2:]
    ext_bshape = B_tensor.shape
    bshape = ext_bshape[2:]
    b_pad_shape = torch.tensor(A_tensor.shape, dtype=torch.long) - torch.tensor(B_tensor.shape, dtype=torch.long)
    ext_valid_shape = b_pad_shape + 1
    batched_valid_shape = ext_valid_shape + torch.tensor([8 - 1, 0, 0, 0])
    valid_shape = ext_valid_shape[2:]

    M_A_FFT = corr_target_setup(M_A)

    A_ffts = []
    for a in range(Q_A):
        A_ffts.append(corr_target_setup(float_compare(A_tensor, a)))

    del A_tensor
    del M_A

    if normalize_mi:
        H_MARG = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
        H_AB = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
    else:
        MI = create_float_tensor(ext_valid_shape, on_gpu, 0.0)



    # preprocess B for angle
    B_tensor_rotated = B_tensor

    M_B_rotated = M_B
    # B_tensor_rotated = torch.round(M_B_rotated * B_tensor_rotated + (1 - M_B_rotated) * (Q_B + 1))
    B_tensor_rotated = ste_round(M_B_rotated * B_tensor_rotated + (1 - M_B_rotated) * (Q_B + 1))

    B_tensor_rotated = F.pad(B_tensor_rotated,
                             (0, ext_ashape[-1] - ext_bshape[-1], 0, ext_ashape[-2] - ext_bshape[-2], 0, 0, 0, 0),
                             mode='constant', value=Q_B + 1)
    M_B_rotated = F.pad(M_B_rotated,
                        (0, ext_ashape[-1] - ext_bshape[-1], 0, ext_ashape[-2] - ext_bshape[-2], 0, 0, 0, 0),
                        mode='constant', value=0)


    M_B_FFT = corr_template_setup(M_B_rotated)
    del M_B_rotated

    N = torch.clamp(corr_apply(M_A_FFT, M_B_FFT, ext_valid_shape), min=eps, max=None)

    b_ffts = fft_of_levelsets(B_tensor_rotated, Q_B, Q_B, corr_template_setup)

    for bext in range(len(b_ffts)):
        b_fft = b_ffts[bext]
        E_M = torch.sum(compute_entropy(corr_apply(M_A_FFT, b_fft[0], batched_valid_shape), N, eps), dim=0)
        if normalize_mi:
            H_MARG = torch.sub(H_MARG, E_M)
        else:
            MI = torch.sub(MI, E_M)
        del E_M

        for a in range(Q_A):
            A_fft_cuda = A_ffts[a]

            if bext == 0:
                E_M = compute_entropy(corr_apply(A_fft_cuda, M_B_FFT, ext_valid_shape), N, eps)
                if normalize_mi:
                    H_MARG = torch.sub(H_MARG, E_M)
                else:
                    MI = torch.sub(MI, E_M)
                del E_M
            E_J = torch.sum(compute_entropy(corr_apply(A_fft_cuda, b_fft[0], batched_valid_shape), N, eps), dim=0)
            if normalize_mi:
                H_AB = torch.sub(H_AB, E_J)
            else:
                MI = torch.add(MI, E_J)
            del E_J
            del A_fft_cuda
        del b_fft
        if bext == 0:
            del M_B_FFT

        del B_tensor_rotated

        if normalize_mi:
            MI = torch.clamp((H_MARG / (H_AB + eps) - 1), 0.0, 1.0)


        (max_n, _) = torch.max(torch.reshape(N, (-1,)), 0)
        N_filt = torch.lt(N, overlap * max_n)
        MI[N_filt] = 0.0
        del N_filt, N

        MI_vec = torch.reshape(MI, (-1,))
        (val, ind) = torch.max(MI_vec, -1)
        return val



if __name__ == "__main__":
    # Generate a 2D image of random labels ranging from 1 to 8
    image_size = (100, 100)  # Define the size of the image
    num_labels = 8  # Number of labels (1-8)

    # Generate random labels
    random_labels_image = np.random.randint(1, num_labels + 1, size=image_size)
    random_labels_tensor = torch.tensor(random_labels_image).reshape(1, 1, 100, 100).float().cuda()
    random_labels_image_1 = np.random.randint(1, num_labels + 1, size=image_size)
    random_labels_tensor_1 = torch.tensor(random_labels_image_1).reshape(1, 1, 100, 100).float().cuda()


    val, ind = mi_loss(random_labels_tensor, random_labels_tensor_1)
    print(val, ind)
    #
    # # results.append((ang, val, ind))
    #
    # if normalize_mi:
    #     H_MARG.fill_(0)
    #     H_AB.fill_(0)
    # else:
    #     MI.fill_(0)

