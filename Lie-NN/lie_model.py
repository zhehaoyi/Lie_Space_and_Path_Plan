import torch


class liemodel():
    def __init__(self, init_vector):
        super(liemodel, self).__init__()
        self.init_vector = init_vector

    # the target lie algebra is so(3)
    # transfer a random vector to lie algebra so(3)
    def init_vector_or_eigenvalues_to_so3(self, shcema, eigenvalue):
        if shcema == 'original':
            v1, v2, v3 = self.init_vector.unbind()
            init_vector_so3 = torch.tensor([
                [0, -v3, v2],
                [v3, 0, -v1],
                [-v2, v1, 0]
            ], dtype=self.init_vector.dtype, device=self.init_vector.device)
            return init_vector_so3
        if shcema == 'feature vector':
            v1, v2, v3 = eigenvalue.unbind()
            eigenvalue_so3 = torch.tensor([
                [0, -v3, v2],
                [v3, 0, -v1],
                [-v2, v1, 0]
            ], dtype=eigenvalue.dtype, device=eigenvalue.device)
            return eigenvalue_so3

    # enhance the Lie group R
    def init_vector_so3_eigenvalus(self, init_vector_so3):
        eigenvalues, _ = torch.linalg.eig(init_vector_so3)
        return eigenvalues

    # transfer a Lie algebra so(3) to Lie group SO(3)==>R
    def so3_to_SO3(self):
        rotation_theta_init_vector_so3 = torch.norm(self.init_vector, dim=-1, keepdim=True)
        init_vector_so3 = self.init_vector_or_eigenvalues_to_so3('original', 0)

        init_vector_so3_eigenvalus = self.init_vector_so3_eigenvalus(init_vector_so3)
        rotation_theta_eigenvalus_so3 = torch.norm(init_vector_so3_eigenvalus, dim=-1, keepdim=True)
        eigenvalues_so3 = self.init_vector_or_eigenvalues_to_so3('feature vector', init_vector_so3_eigenvalus)

        # transfer init vector so3 to SO3
        sin_theta_init_vector_so3 = torch.sin(rotation_theta_init_vector_so3)
        cos_theta_init_vector_so3 = torch.cos(rotation_theta_init_vector_so3)
        R_init_vector_SO3 = (torch.eye(3, dtype=self.init_vector.dtype, device=self.init_vector.device) + (
                sin_theta_init_vector_so3 / rotation_theta_init_vector_so3) * init_vector_so3 + (
                                     (1 - cos_theta_init_vector_so3) / (
                                 rotation_theta_init_vector_so3) ** 2) * torch.mm(
            init_vector_so3, init_vector_so3))

        # transfer eigenvalus to SO3
        sin_theta_eigenvalues_so3 = torch.sin(rotation_theta_eigenvalus_so3)
        cos_theta_eigenvalues_so3 = torch.cos(rotation_theta_eigenvalus_so3)
        R_eigenvalues_SO3 = (torch.eye(3, dtype=self.init_vector.dtype, device=self.init_vector.device) + (
                sin_theta_eigenvalues_so3 / rotation_theta_eigenvalus_so3) * eigenvalues_so3 + (
                                     (1 - cos_theta_eigenvalues_so3) / (
                                 rotation_theta_eigenvalus_so3) ** 2) * torch.mm(
            eigenvalues_so3, eigenvalues_so3))

        return R_init_vector_SO3 + R_eigenvalues_SO3  # return enhanced SO3

    def enhanced_SO3_to_so3(self):
        eps = 1e-5
        enhanced_SO3 = self.so3_to_SO3()
        tr_R = enhanced_SO3.trace()
        tr_R = torch.abs(tr_R)
        theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + eps, 1 - eps))
        sin_theta = torch.sin(theta)
        if sin_theta < eps:
            # 0 rotation or pi rotation
            if tr_R > 3 - eps:  # θ ≈ 0
                return torch.zeros(3, dtype=enhanced_SO3.dtype, device=enhanced_SO3.device)
            else:  # θ ≈ π
                # compute(R + I)'szero space
                A = enhanced_SO3 + torch.eye(3, dtype=enhanced_SO3.dtype, device=enhanced_SO3.device)
                _, _, V = torch.linalg.svd(A)
                v = V[:, -1]  # the minimized singular's vector
                return theta * v
        else:
            # General
            v = torch.stack([
                enhanced_SO3[2, 1] - enhanced_SO3[1, 2],
                enhanced_SO3[0, 2] - enhanced_SO3[2, 0],
                enhanced_SO3[1, 0] - enhanced_SO3[0, 1]
            ]) / (2 * sin_theta)
            return theta * v

    def final_vector(self):
        enhanced_vector = self.enhanced_SO3_to_so3()
        return 0.9 * enhanced_vector + 0.1 * self.init_vector
