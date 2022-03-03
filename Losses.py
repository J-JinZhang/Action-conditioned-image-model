"""
Author: Dr. Jin Zhang 
E-mail: j.zhang.vision@gmail.com
Created on 2022.02.06
"""

import torch


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps):
        preds, _, _ = self.dis(fake_samps)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class WGAN_GP(GANLoss):

    def __init__(self, dis, drift=0.001, use_gp=False):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad = True

        # forward pass
        op = self.dis(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps)
        real_out = self.dis(real_samps)

        loss = (torch.mean(fake_out) - torch.mean(real_out)
                + (self.drift * torch.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps):
        # calculate the WGAN loss for generator
        loss = -torch.mean(self.dis(fake_samps))

        return loss


class LSGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        return 0.5 * (((torch.mean(self.dis(real_samps)) - 1) ** 2)
                      + (torch.mean(self.dis(fake_samps))) ** 2)

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((torch.mean(self.dis(fake_samps)) - 1) ** 2)


class LSGAN_SIGMOID(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        from torch.nn.functional import sigmoid
        real_scores = torch.mean(sigmoid(self.dis(real_samps)))
        fake_scores = torch.mean(sigmoid(self.dis(fake_samps)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps):
        from torch.nn.functional import sigmoid
        scores = torch.mean(sigmoid(self.dis(fake_samps)))
        return 0.5 * ((scores - 1) ** 2)


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        r_preds, r_mus, r_sigmas = self.dis(real_samps)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps)

        loss = (torch.mean(torch.nn.ReLU()(1 - r_preds)) +
                torch.mean(torch.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps):
        return -torch.mean(self.dis(fake_samps))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(torch.nn.ReLU()(1 - r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)
        #print(f"r_preds: {r_preds}   f_preds: {f_preds}")

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)
        #print(f"r_f_diff: {r_f_diff.data}    f_r_diff: {f_r_diff.data}")

        # return the loss
        return (torch.mean(torch.nn.ReLU()(1 + r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 - f_r_diff)))

    
class Relativistic_Average_LSGAN(GANLoss):
    
    def __init__(self, dis):
        super().__init__(dis)
        self.real_label = 0.9
        self.fake_label = 0.0
    
    def dis_loss(self, real_samps, fake_samps):
        
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)
        
        real_tensor = torch.FloatTensor(r_preds.size()).fill_(self.real_label)
        real_tensor.requires_grad = False
        target_tensor = real_tensor.cuda()
        
        return (torch.mean((r_preds - torch.mean(f_preds) - target_tensor)** 2)
               + torch.mean((f_preds - torch.mean(r_preds) + target_tensor)** 2))/2
        
    def gen_loss(self, real_samps, fake_samps):
        
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)
        
        real_tensor = torch.FloatTensor(r_preds.size()).fill_(self.real_label)
        real_tensor.requires_grad = False
        target_tensor = real_tensor.cuda()
        
        return (torch.mean((r_preds - torch.mean(f_preds) + target_tensor)** 2)
               + torch.mean((f_preds - torch.mean(r_preds) - target_tensor)** 2))/2
    
    
class PixelwiseL1():
    def __init__(self):
        
        self.loss_func = torch.nn.SmoothL1Loss()
        self.loss_func = self.loss_func.cuda()
    
    def gen_loss(self, real_samps, fake_samps):
        loss = torch.zeros(7).cuda()
        for i, (real, fake) in enumerate(zip(real_samps, fake_samps)):
            #print('size of fake_samps-6 is: {}'.format(fake_samps[6].size()))
            #loss.append((i+1)/2*self.loss_func(real, fake))
            loss[i] = (i+1)*self.loss_func(real, fake)
        #print('loss is: {}'.format(loss))
        return torch.sum(loss)
    
    
class Perceptual():
    def __init__(self, feats_model):
        self.feats_model = feats_model
        self.loss_func = torch.nn.MSELoss()
        self.loss_func = self.loss_func.cuda()
    
    def gen_loss(self, real_samps, fake_samps):
        # multi-scale
        """loss = torch.zeros(7).cuda()
        for i, (real, fake) in enumerate(zip(real_samps, fake_samps)):
            real_feats = self.feats_model(real)
            fake_feats = self.feats_model(fake)
            loss[i] = (i+1)*self.loss_func(fake_feats, real_feats)
            loss_val = torch.sum(loss)"""
            
        # single-scale
        real_feats = self.feats_model(real_samps)
        fake_feats = self.feats_model(fake_samps)
        loss_val = self.loss_func(fake_feats, real_feats)
        return loss_val
    
    
