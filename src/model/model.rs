use linfa::prelude::*;
use linfa_linear::FittedLinearRegression;
use linfa_linear::LinearRegression;
use ndarray::Ix1;
use std::error::Error;

#[derive(Debug)]
pub struct ModelX<'a> {
    pub id: uuid::Uuid,
    pub model_type: ModelsType,
    pub model_fitted: Option<FittedLinearRegression<f64>>,
    pub dataset: &'a Dataset<f64, f64, Ix1>,
}

#[derive(Debug)]
pub enum ModelsType {
    LinearRegression(LinearRegression),
}

impl<'a> ModelX<'a> {
    /// Create a new model with dataset and model type
    pub fn new(model_type: ModelsType, dataset: &'a Dataset<f64, f64, Ix1>) -> Self {
        let m = ModelX {
            id: uuid::Uuid::new_v4(),
            model_type,
            model_fitted: None,
            dataset,
        };
        println!(
            "{}",
            format!(
                "Model id: {}, \nModel Type: {:?} \nDataset Samples: {:?}",
                m.id,
                m.model_type,
                m.dataset.nsamples()
            )
        );
        m
    }

    /// Train the model with the dataset
    /// If the training is successful, the model_fitted field is filled
    /// with the fitted model can be used for prediction
    pub fn train(&mut self) -> Result<(), Box<dyn Error>> {
        println!(
            "Training Model with {:?}\nDataset Samples: {}",
            self.model_type,
            self.dataset.nsamples()
        );
        match &self.model_type {
            ModelsType::LinearRegression(model) => {
                let model_fitted = model.fit(&self.dataset)?;
                self.model_fitted = Some(model_fitted);
            }
        }
        Ok(())
    }
}
