import torch

def cartesian_to_polar(real, imag):
    """
    Convert cartesian fields to polar fields
    
    Returns:
    - mag: Magnitude of the field
    - phase: Phase of the field
    """
    complex = torch.complex(real, imag)
    mag = torch.abs(complex)
    phase = torch.angle(complex)
    return mag, phase

def polar_to_cartesian(mag, phase):
    """
    Convert polar fields to cartesian fields
    
    Returns:
    - real: Real part of the field
    - imag: Imaginary part of the field
    """
    complex = mag * torch.cos(phase) + 1j * mag * torch.sin(phase)
    # separate into real and imaginary
    real = torch.real(complex)
    imag = torch.imag(complex)
    return real, imag