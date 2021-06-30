#!/usr/bin/env python3

import torch


class NeuralLuSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, A_lu, At_lu):
        ctx.At_lu = At_lu  # keep lu factorization of A-transpose
        u = torch.lu_solve(b, *A_lu)
        return u

    @staticmethod
    def backward(ctx, du):
        At_lu = ctx.At_lu
        db = torch.lu_solve(du, *At_lu)
        return db, None, None


class NeuralSparseLuSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, A_lu, At_lu):
        ctx.At_lu = At_lu  # keep lu factorization of A-transpose

        all_u = []
        for i in range(len(A_lu)):
            curr_u = A_lu[i].solve(b[i].detach().numpy())
            all_u.append(curr_u)
        
        u = torch.stack([torch.from_numpy(x) for x in all_u])
        u = u.to(b.device)
        return u

    @staticmethod
    def backward(ctx, du):
        At_lu = ctx.At_lu
        all_db = []
        for i in range(len(At_lu)):
            curr_db = At_lu[i].solve(du[i].detach().numpy())
            all_db.append(curr_db)

        db = torch.stack([torch.from_numpy(x) for x in all_db])
        db = db.to(du.device)
        return db, None, None

