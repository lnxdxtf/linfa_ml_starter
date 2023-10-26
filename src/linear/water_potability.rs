use std::error::Error;

use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ml_linfa_test::{build_dataset, fill_empty_fields, plot};

const INPUT_PATH: &str = "datasets/water_potability.csv";
const OUTPUT_PATH: &str = "datasets/processed/water_potability_output.csv";
const DEFAULT_VALUE_MISSING: &str = "0.0";

fn pre_process_csv() -> Result<(), Box<dyn Error>> {
    fill_empty_fields(INPUT_PATH, OUTPUT_PATH, DEFAULT_VALUE_MISSING)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    pre_process_csv()?;
    let feature_names = vec![
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
        "Potability",
    ];

    let dataset = build_dataset(OUTPUT_PATH, feature_names.clone())?;
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset)?;
    println!("intercept:  {}", model.intercept());
    println!("parameters: {}", model.params());

    // println!("{:?}", dataset.records);

    plot(
        "water_potability/test",
        &dataset,
        feature_names[0],
        feature_names[9],
    )?;
    Ok(())
}
