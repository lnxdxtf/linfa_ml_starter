use std::error::Error;
mod dataframe;
mod model;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use model::model::*;

fn main() -> Result<(), Box<dyn Error>> {
    let path = "datasets/water_potability.csv";
    let column_target_idx = 0;
    let train_ratio = 0.9;

    let (dataset_train, dataset_test) = dataframe::dataframe::get_dataframe_polars_linfa(
        path,
        true,
        b',',
        polars::prelude::FillNullStrategy::Mean,
        train_ratio,
        column_target_idx,
    )?;

    let rgs_model = LinearRegression::new();
    let mut model = ModelX::new(ModelsType::LinearRegression(rgs_model), &dataset_train);
    model.train()?;
    if let Some(model_fitted) = model.model_fitted {
        println!("Predicting with Dataset: {}", &dataset_test.nsamples());
        let predictions = model_fitted.predict(&dataset_test);
        println!("{:?}", predictions);
    }

    Ok(())
}
