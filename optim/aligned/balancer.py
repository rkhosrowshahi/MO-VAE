import torch

from .solver import ProcrustesSolver
from .. import basic_balancer
from .. import balancers


@balancers.register("amtl")
class AlignedMTLBalancer(basic_balancer.BasicBalancer):
    def __init__(self, scale_mode="min", scale_decoder_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.scale_decoder_grad = scale_decoder_grad
        self.scale_mode = scale_mode
        print("AMGDA balancer scale mode:", self.scale_mode)

    def step_with_model(
        self, data: torch.Tensor, model: torch.nn.Module, criteria: dict, **kwargs
    ) -> None:
        losses, hrepr = self.compute_losses(data, model, criteria)
        self.step(
            losses=losses,
            shared_params=list(model.encoder.parameters()),
            task_specific_params={
                "reconstruction": model.decoder.parameters(),
                "kl": model.mu.parameters(),
                "kl": model.log_var.parameters(),
            },
            shared_representation=[model.mu.parameters(), model.log_var.parameters()],
            last_shared_layer_params=None,
            model=model,
        )

    def step(
        self,
        losses,
        shared_params,
        task_specific_params,
        shared_representation=None,
        last_shared_layer_params=None,
        model=None,
    ):
        print(type(shared_params))
        grads = self.get_G_wrt_shared2(
            losses, shared_params, shared_representation, update_decoder_grads=True
        )
        grads, weights, singulars = ProcrustesSolver.apply(
            grads.T.unsqueeze(0), self.scale_mode
        )
        grad, weights = grads[0].sum(-1), weights.sum(-1)

        if self.compute_stats:
            self.compute_metrics(grads[0])

        self.set_shared_grad(shared_params, grad)

        if self.scale_decoder_grad is True:
            self.scale_task_specific_params(
                task_specific_params,
                weights={task_id: weights[i] for i, task_id in enumerate(losses)},
            )
            # self.apply_decoder_scaling(task_specific_params, weights)

        # self.set_losses({task_id: losses[task_id] * weights[i] for i, task_id in enumerate(losses)})
        # self.zero_grad_model(model)
        # total_loss = sum(
        #     losses[task_id] * weights[i] for i, task_id in enumerate(losses)
        # )
        # total_loss.backward()

        self.set_loss_weights({task_id: weights[i] for i, task_id in enumerate(losses)})
        self.set_losses(losses)


@balancers.register("amtlub")
class AlignedMTLUBBalancer(basic_balancer.BasicBalancer):
    def __init__(self, scale_decoder_grad=False, scale_mode="min", **kwargs):
        super().__init__(**kwargs)
        self.scale_decoder_grad = scale_decoder_grad
        self.scale_mode = scale_mode

    def step_with_model(self, data, targets, model, criteria, **kwargs):
        self.zero_grad_model(model)
        hrepr = model.encoder(data)

        grads, losses = self.get_model_G_wrt_hrepr(
            hrepr,
            targets,
            model,
            criteria,
            update_decoder_grads=True,
            return_losses=True,
        )
        grads, weights, singulars = ProcrustesSolver.apply(
            grads.T.unsqueeze(0), self.scale_mode
        )
        grad, weights = grads[0].sum(-1), weights.sum(-1)

        if self.compute_stats:
            # Computationally expensive, use it for demonstration only.
            # This code is not supposed to be executed within real training pipelines.
            wgrads = list()
            for t in range(grads.shape[-1]):
                hrepr.backward(grads[:, :, t].view_as(hrepr), retain_graph=True)

                wgrads.append(
                    torch.cat(
                        [
                            p.grad.flatten().detach().data.clone()
                            for p in model.encoder.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                model.encoder.zero_grad()

            wgrads = torch.stack(wgrads, dim=-1)
            self.compute_metrics(wgrads)

        grad, weights = grads.sum(-1).view_as(hrepr), weights.sum(-1)
        hrepr.backward(grad)

        self.set_losses(losses)
        if self.scale_decoder_grad is True:
            self.apply_decoder_scaling(model.decoders, weights)
