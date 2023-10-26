use linfa::prelude::*;
use linfa_linear::LinearRegression;
use linfa_logistic::LogisticRegression;
use ml_linfa_test::{build_dataset, fill_empty_fields, plot};
use std::error::Error;

const INPUT_PATH: &str = "datasets/water_potability.csv";
const OUTPUT_PATH: &str = "datasets/processed/water_potability_output.csv";
const DEFAULT_VALUE_MISSING: &str = "0.0";

fn pre_process_csv() -> Result<(), Box<dyn Error>> {
    fill_empty_fields(INPUT_PATH, OUTPUT_PATH, DEFAULT_VALUE_MISSING)?;
    Ok(())
}

// Need run other model, like a regression logistic
// Bad idea use linear regression for this dataset
// the values are too close to 0


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
    println!(
        "Fit Linear Regression Water Potability with #{}",
        dataset.nsamples()
    );
    let model = lin_reg.fit(&dataset)?;
    let m_intercept = model.intercept();
    println!("intercept:  {}", m_intercept);
    let m_params = model.params();
    println!("parameters: {}", m_params);
    let m_predict = model.predict(&dataset);
    println!("predict: {}", m_predict);

    plot(
        "water_potability/dataset_train",
        &dataset,
        feature_names[0],
        feature_names[9],
    )?;

    Ok(())
}
