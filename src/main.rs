use std::error::Error;
mod dataframe;
mod model;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use model::model::*;

fn main() -> Result<(), Box<dyn Error>> {
    let path = "datasets/water_potability.csv";
    let column_target_idx = 0;

    let dataset = dataframe::dataframe::get_dataframe_polars_linfa(
        path,
        true,
        b',',
        polars::prelude::FillNullStrategy::Mean,
        column_target_idx,
    )?;

    let rgs_model = LinearRegression::new();
    let mut model = ModelX::new(ModelsType::LinearRegression(rgs_model), &dataset);
    model.train()?;
    if let Some(model_fitted) = model.model_fitted {
        let predictions = model_fitted.predict(&dataset);
        println!("{:?}", predictions);
    }

    Ok(())
}
