"""
Tests for gfram.models module
"""

import pytest
import torch
import numpy as np


class TestGeometricTransformer:
    """Tests for GeometricTransformer model"""
    
    def test_model_creation(self):
        """Test model can be created"""
        from gfram.models import create_geometric_transformer
        
        model = create_geometric_transformer(config_name='tiny')
        assert model is not None
    
    def test_model_configs(self):
        """Test different model configurations"""
        from gfram.models import create_geometric_transformer
        
        configs = ['tiny', 'small', 'base', 'large']
        
        for config in configs:
            model = create_geometric_transformer(config_name=config)
            assert model is not None
    
    def test_invalid_config(self):
        """Test error on invalid config"""
        from gfram.models import create_geometric_transformer
        
        with pytest.raises(ValueError):
            create_geometric_transformer(config_name='invalid')
    
    def test_forward_pass(self, batch_landmarks, transformer_model):
        """Test forward pass works"""
        output = transformer_model(batch_landmarks)
        
        assert output is not None
        assert output.shape[0] == batch_landmarks.shape[0]
    
    def test_output_dimension(self):
        """Test output dimension matches config"""
        from gfram.models import create_geometric_transformer
        
        model = create_geometric_transformer(config_name='tiny')
        
        landmarks = torch.randn(2, 478, 3)
        output = model(landmarks)
        
        assert output.shape == (2, 128)  # tiny config has output_dim=128
    
    def test_model_478_landmarks(self):
        """Test model with 478 landmarks"""
        from gfram.models import create_geometric_transformer
        
        model = create_geometric_transformer(num_landmarks=478)
        landmarks = torch.randn(1, 478, 3)
        
        output = model(landmarks)
        assert output is not None
    
    def test_model_468_landmarks(self):
        """Test model with 468 landmarks"""
        from gfram.models import create_geometric_transformer
        
        model = create_geometric_transformer(num_landmarks=468)
        landmarks = torch.randn(1, 468, 3)
        
        output = model(landmarks)
        assert output is not None
    
    def test_model_flexible_landmarks(self):
        """Test model handles different landmark counts"""
        from gfram.models import create_geometric_transformer
        
        # Model expects 478
        model = create_geometric_transformer(num_landmarks=478)
        
        # But receives 468 (should be padded)
        landmarks = torch.randn(1, 468, 3)
        output = model(landmarks)
        
        assert output is not None
    
    def test_extract_embedding(self, batch_landmarks):
        """Test extract_embedding method"""
        from gfram.models import create_geometric_transformer
        
        model = create_geometric_transformer(config_name='tiny')
        model.eval()
        
        embedding = model.extract_embedding(batch_landmarks)
        
        assert embedding is not None
        assert embedding.shape[0] == batch_landmarks.shape[0]
    
    def test_model_eval_mode(self, transformer_model, batch_landmarks):
        """Test model in eval mode produces consistent results"""
        transformer_model.eval()
        
        with torch.no_grad():
            output1 = transformer_model(batch_landmarks)
            output2 = transformer_model(batch_landmarks)
        
        torch.testing.assert_close(output1, output2)
    
    def test_model_parameters(self):
        """Test model has trainable parameters"""
        from gfram.models import create_geometric_transformer, get_model_info
        
        model = create_geometric_transformer(config_name='base')
        info = get_model_info(model)
        
        assert info['total_params'] > 0
        assert info['trainable_params'] > 0
        assert info['model_size_mb'] > 0


class TestModelInfo:
    """Tests for get_model_info function"""
    
    def test_get_model_info(self):
        """Test get_model_info returns correct info"""
        from gfram.models import create_geometric_transformer, get_model_info
        
        model = create_geometric_transformer(config_name='tiny')
        info = get_model_info(model)
        
        assert 'num_landmarks' in info
        assert 'embed_dim' in info
        assert 'total_params' in info
        assert 'model_size_mb' in info
        
        assert info['num_landmarks'] == 478
        assert info['embed_dim'] == 128  # tiny config


class TestLossFunctions:
    """Tests for loss functions"""
    
    def test_arcface_loss_import(self):
        """Test ArcFace loss can be imported"""
        from gfram.models.losses import ArcFaceLoss
        
        loss_fn = ArcFaceLoss(in_features=256, out_features=100)
        assert loss_fn is not None
    
    def test_triplet_loss_import(self):
        """Test Triplet loss can be imported"""
        from gfram.models.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=0.2)
        assert loss_fn is not None
    
    def test_arcface_forward(self):
        """Test ArcFace loss forward pass"""
        from gfram.models.losses import ArcFaceLoss
        
        loss_fn = ArcFaceLoss(in_features=256, out_features=10)
        
        embeddings = torch.randn(4, 256)
        labels = torch.tensor([0, 1, 2, 3])
        
        output = loss_fn(embeddings, labels)
        assert output is not None
    
    def test_triplet_forward(self):
        """Test Triplet loss forward pass"""
        from gfram.models.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=0.2)
        
        anchor = torch.randn(4, 256)
        positive = torch.randn(4, 256)
        negative = torch.randn(4, 256)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss is not None
        assert loss.item() >= 0


class TestPositionalEncoding:
    """Tests for positional encoding"""
    
    def test_positional_encoding(self):
        """Test positional encoding"""
        from gfram.models.geometric_transformer import PositionalEncoding
        
        pe = PositionalEncoding(d_model=256, max_len=500)
        x = torch.randn(2, 478, 256)
        
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_geometric_embedding(self):
        """Test geometric embedding layer"""
        from gfram.models.geometric_transformer import GeometricEmbedding
        
        embed = GeometricEmbedding(input_dim=3, embed_dim=256)
        x = torch.randn(2, 478, 3)
        
        output = embed(x)
        
        assert output.shape == (2, 478, 256)
