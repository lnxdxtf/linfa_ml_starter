use csv::{ReaderBuilder, StringRecord, Trim, WriterBuilder};
use linfa::Dataset;
use linfa_datasets::array_from_csv;
use ndarray::{s, Array1, Ix1};
use plotters::{prelude::*, style::full_palette::PURPLE_500};
use std::error::Error;

pub fn fill_empty_fields(
    input_path: &str,
    output_path: &str,
    default_value: &str,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().trim(Trim::All).from_path(input_path)?;
    let headers = rdr.headers()?.clone();

    let mut wtr = WriterBuilder::new().from_path(output_path)?;

    wtr.write_record(&headers)?;

    for result in rdr.records() {
        let record = result?;
        let mut new_record = StringRecord::new();
        for field in record.iter() {
            if field.is_empty() {
                new_record.push_field(default_value);
            } else {
                if let Ok(f) = field.parse::<i32>() {
                    let field_f64_string = format!("{:.1}", f as f64);
                    new_record.push_field(&field_f64_string);
                } else {
                    new_record.push_field(field);
                }
            }
        }
        wtr.write_record(&new_record)?;
    }

    wtr.flush()?;
    Ok(())
}

pub fn build_dataset(
    file: &str,
    feature_names: Vec<&str>,
) -> Result<Dataset<f64, f64, Ix1>, Box<dyn Error>> {
    let arr = array_from_csv(std::fs::File::open(file)?, true, b',')?;
    let (data, targets) = (
        arr.slice(s![.., 0..feature_names.len()]).to_owned(),
        arr.column(feature_names.len() - 1).to_owned(),
    );
    let dt = Dataset::new(data, targets).with_feature_names(feature_names.clone());
    println!("Dataset shape dim: {:?}", dt.records.dim());
    println!(
        "Features names: {:?} | len: {:?}",
        dt.feature_names(),
        feature_names.len()
    );
    Ok(dt)
}
pub fn plot(
    plot_output: &str,
    dataset: &Dataset<f64, f64, Ix1>,
    feature_x: &str,
    feature_y: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = format!("plots/{}.png", plot_output);
    let root = BitMapBackend::new(output.as_str(), (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let caption = format!("{} vs {}", feature_x, feature_y);
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-30f64..30f64, -30f64..30f64)?;

    chart
        .configure_mesh()
        .x_desc(feature_x)
        .y_desc(feature_y)
        .draw()?;
    let idx_feature_x = dataset
        .feature_names()
        .iter()
        .position(|ft_name| ft_name == feature_x)
        .unwrap();
    let idx_feature_y = dataset
        .feature_names()
        .iter()
        .position(|ft_name| ft_name == feature_y)
        .unwrap();

    let axis_x_values: Array1<f64> = dataset.records.column(idx_feature_x).to_owned();
    let axis_y_values: Array1<f64> = dataset.records.column(idx_feature_y).to_owned();

    // let line_data: Vec<(f64, f64)> = axis_x_values
    //     .iter()
    //     .zip(axis_y_values.iter())
    //     .map(|(&x, &y)| (x, y))
    //     .collect();

    // chart.draw_series(LineSeries::new(line_data, &RED))?;

    chart.draw_series(PointSeries::of_element(
        axis_x_values.into_iter().zip(axis_y_values.into_iter()),
        1,
        &PURPLE_500,
        &|c, s, st| {
            return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
        },
    ))?;

    root.present()?;
    Ok(())
}
