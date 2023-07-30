"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """


    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.
    clip_predicted_rating: boolean, optional.
        If true, clips rating predictions to [0,5] using sigmoid activation followed by scaling.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True, clip_predicted_rating=True):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing
        self.clip_predicted_rating = clip_predicted_rating

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        if self.embedding_sharing:
            self.U = ScaledEmbedding(num_users, self.embedding_dim)
            self.Q = ScaledEmbedding(num_items, self.embedding_dim)
        else:
            # for matrix factorization
            self.U1 = ScaledEmbedding(num_users, self.embedding_dim)
            self.Q1 = ScaledEmbedding(num_items, self.embedding_dim)

            # for regression model: rating prediction
            self.U2 = ScaledEmbedding(num_users, self.embedding_dim)
            self.Q2 = ScaledEmbedding(num_items, self.embedding_dim)

        # for matrix factorization prediction
        self.A = ZeroEmbedding(num_users, 1)
        self.B = ZeroEmbedding(num_items, 1)

        # layers for regression task MLP
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layer_sizes[-1], 1))
        if self.clip_predicted_rating:
            self.layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*self.layers)
        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        if self.embedding_sharing:
            batch_user_embeddings = self.U(user_ids)
            batch_item_embeddings = self.Q(item_ids)
            elem_wise_mul = batch_user_embeddings * batch_item_embeddings

            ## shape: (batch_size)
            predictions = torch.sum(elem_wise_mul, dim = 1) + \
                    torch.squeeze(self.A(user_ids), 1) + \
                    torch.squeeze(self.B(item_ids), 1)

            # shape = [batch_size, 3d]
            regression_input = torch.concat([batch_user_embeddings, batch_item_embeddings, elem_wise_mul], dim = 1)
        else:

            batch_user_embeddings_MF = self.U1(user_ids)
            batch_item_embeddings_MF = self.Q1(item_ids)

            batch_user_embeddings_reg = self.U2(user_ids)
            batch_item_embeddings_reg = self.Q2(item_ids)

            elem_wise_mul_MF = batch_user_embeddings_MF * batch_item_embeddings_MF
            elem_wise_mul_reg = batch_user_embeddings_reg * batch_item_embeddings_reg

            ## shape: (batch_size)
            predictions = torch.sum(elem_wise_mul_MF, dim=1) + \
                    torch.squeeze(self.A(user_ids), 1) + \
                    torch.squeeze(self.B(item_ids), 1)

            # shape = [batch_size, 3d]
            regression_input = torch.concat([batch_user_embeddings_reg, batch_item_embeddings_reg, elem_wise_mul_reg], dim=1)

        # shape = (batch_size)
        score = torch.squeeze(self.mlp(regression_input), dim=1)
        if self.clip_predicted_rating:
            score = 5.0 * score  # Convert [0,1] probabilities back to [0,5] rating score.

        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score