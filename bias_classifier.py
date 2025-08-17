import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import json

class BiasClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        
    def create_sample_data(self):
        """Create sample training data for bias classification (non-human subjects)"""
        # Sample biased and unbiased text examples
        texts = [
            # Biased examples (label: 1)
            "All electric cars are unreliable and break down frequently",
            "Apple products are always overpriced and inferior to alternatives",
            "Free software is always buggy and poorly designed",
            "Organic food is just a marketing scam with no real benefits",
            "Classical music is boring and only for pretentious people",
            "Video games cause violence and are a waste of time",
            "Social media platforms are destroying society completely",
            "Modern art is meaningless and anyone could create it",
            
            # Unbiased examples (label: 0)
            "Electric vehicle reliability varies by manufacturer and model specifications",
            "Product pricing reflects various factors including features, materials, and market positioning",
            "Software quality depends on development resources, testing, and maintenance practices",
            "Organic farming methods have specific benefits and limitations worth considering",
            "Musical preferences vary widely and reflect personal taste and cultural background",
            "Video games have diverse content and can offer various educational and entertainment benefits",
            "Social media platforms have both positive and negative impacts on communication",
            "Modern art encompasses diverse styles and techniques with varying levels of complexity",
            "Technology adoption involves weighing costs, benefits, and specific use cases",
            "Food choices depend on personal preferences, health needs, and availability",
        ]
        
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        return texts, labels
    
    def prepare_dataset(self, texts, labels):
        """Prepare dataset for training"""
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
        
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train_model(self):
        """Train the bias classification model"""
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        
        # Create sample data
        texts, labels = self.create_sample_data()
        
        # Prepare dataset
        dataset = self.prepare_dataset(texts, labels)
        
        # Split dataset (using simple split for demo)
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./bias_classifier_results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model('./bias_classifier_model')
        self.tokenizer.save_pretrained('./bias_classifier_model')
        
        print("Model training completed and saved!")
    
    def load_model(self, model_path='./bias_classifier_model'):
        """Load a trained model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create pipeline for easy inference
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        
        print("Model loaded successfully!")
    
    def predict(self, text):
        """Predict if text contains biased statements about non-human subjects"""
        if self.classifier is None:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        results = self.classifier(text)
        
        # Get the prediction
        prediction = results[0]
        
        # Format results
        bias_score = None
        unbiased_score = None
        
        for result in prediction:
            if result['label'] == 'LABEL_1':  # Biased
                bias_score = result['score']
            else:  # Unbiased
                unbiased_score = result['score']
        
        is_biased = bias_score > unbiased_score
        confidence = max(bias_score, unbiased_score)
        
        return {
            'text': text,
            'is_biased': is_biased,
            'confidence': confidence,
            'bias_probability': bias_score,
            'unbiased_probability': unbiased_score
        }
    
    def batch_predict(self, texts):
        """Predict multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# Usage example
if __name__ == "__main__":
    # Initialize classifier
    classifier = BiasClassifier()
    
    # Train the model (this will take a few minutes)
    print("Training bias classification model for non-human subjects...")
    classifier.train_model()
    
    # Load the trained model
    classifier.load_model()
    
    # Test predictions
    test_texts = [
        "All smartphones are designed to become obsolete within two years",
        "Smartphone longevity varies based on build quality, usage patterns, and software support",
        "Fast food is always unhealthy and has no nutritional value",
        "Fast food nutritional content varies significantly across different menu items and preparation methods"
    ]
    
    print("\nTesting predictions on products/technology/concepts:")
    for text in test_texts:
        result = classifier.predict(text)
        print(f"Text: {result['text']}")
        print(f"Contains bias: {result['is_biased']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 50)