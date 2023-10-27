use linfa::Dataset;
use ndarray::{s, Ix1};
use polars::prelude::*;
use std::error::Error;

pub fn get_dataframe_polars_linfa(
    path: &str,
    has_header: bool,
    separator: u8,
    fill_strategy: FillNullStrategy,
    train_ratio: f32,
    column_target_idx: usize,
) -> Result<(Dataset<f64, f64, Ix1>, Dataset<f64, f64, Ix1>), Box<dyn Error>> {
    let mut dataframe = CsvReader::from_path(path)?
        .has_header(has_header)
        .with_separator(separator)
        .finish()?;
    dataframe = dataframe.fill_null(fill_strategy)?;

    let features = dataframe.get_column_names();

    // Convert polars::Dataframe to ndarray::ArrayBase
    let arr = dataframe.to_ndarray::<Float64Type>(IndexOrder::C)?;

    let (data, targets) = (
        arr.slice(s![.., 0..features.len()]).to_owned(),
        arr.column(column_target_idx).to_owned(),
    );

    let dt = Dataset::new(data, targets).with_feature_names(features);
    let (train_dt, test_dt) = dt.split_with_ratio(train_ratio);

    Ok((train_dt, test_dt))
}
