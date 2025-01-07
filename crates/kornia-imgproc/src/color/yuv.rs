use crate::parallel;
use kornia_image::{Image, ImageError};

/// Convert an RGB image to an YUV image.
///
/// The input image is assumed to have 3 channels in the order R, G, B.
///
/// # Arguments
///
/// * `src` - The input RGB image assumed to have 3 channels.
/// * `dst` - The output YUV image.
///
/// # Returns
///
/// The YUV image with the following channels:
///
/// * Y: The luma channel.
/// * U: The blue-difference chroma channel.
/// * V: The red-difference chroma channel.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::yuv_from_rgb;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///        width: 4,
///        height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let mut yuv = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// yuv_from_rgb(&image, &mut yuv).unwrap();
///
/// assert_eq!(yuv.num_channels(), 3);
/// assert_eq!(yuv.size().width, 4);
/// assert_eq!(yuv.size().height, 5);
/// ```
pub fn yuv_from_rgb(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // compute the YUV values
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        // Normalize the input to the range [0, 1]
        let r = src_pixel[0] / 255.;
        let g = src_pixel[1] / 255.;
        let b = src_pixel[2] / 255.;

        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let u = -0.147 * r - 0.289 * g + 0.436 * b;
        let v = 0.615 * r - 0.515 * g - 0.100 * b;

        dst_pixel[0] = y;
        dst_pixel[1] = u;
        dst_pixel[2] = v;
    });

    Ok(())
}

/// Convert an YUV image to an RGB image.
///
/// The input image is assumed to have 3 channels in the order Y, U, V.
///
/// # Arguments
///
/// * `src` - The input YUV image assumed to have 3 channels.
/// * `dst` - The output RGB image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgb_from_yuv;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///        width: 4,
///        height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let mut rgb = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// rgb_from_yuv(&image, &mut rgb).unwrap();
///
/// assert_eq!(rgb.num_channels(), 3);
/// assert_eq!(rgb.size().width, 4);
/// assert_eq!(rgb.size().height, 5);
/// ```
pub fn rgb_from_yuv(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // compute the YUV values
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        // Normalize the input to the range [0, 1]
        let y = src_pixel[0];
        let u = src_pixel[1];
        let v: f32 = src_pixel[2];

        let r = y + 1.14 * v;
        let g = y - 0.395 * u - 0.581 * v;
        let b = y + 2.032 * u;

        dst_pixel[0] = r;
        dst_pixel[1] = g;
        dst_pixel[2] = b;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn yuv_from_rgb() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 3,
                height: 2,
            },
            vec![
                255.0, 0.0, 0.0,
                0.0, 255.0, 0.0,
                0.0, 0.0, 255.0,
                0.0, 0.0, 0.0,
                255.0, 255.0, 255.0,
                128.0, 128.0, 128.0,
            ],
        )?;

        #[rustfmt::skip]
        let expected = [
            0.299, -0.147, 0.615,
            0.587, -0.289, -0.515,
            0.114, 0.436, -0.100,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.501961, 0.0, 0.0,
        ];

        let mut yuv = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

        super::yuv_from_rgb(&image, &mut yuv)?;

        assert_eq!(yuv.num_channels(), 3);
        assert_eq!(yuv.size().width, 3);
        assert_eq!(yuv.size().height, 2);

        for (a, b) in yuv.as_slice().iter().zip(expected.iter()) {
            // assert!((a - b).abs() < 1e-6f32);
            assert_eq!(a, b);
        }

        Ok(())
    }

    #[test]
    fn rgb_from_yuv() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![
                0.299, -0.147, 0.615,
                0.587, -0.289, -0.515,
            ],
        )?;

        #[rustfmt::skip]
        let expected = [
            255.0, 0.0, 0.0,
            0.0, 255.0, 0.0,
        ];

        let mut rgb = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

        super::rgb_from_yuv(&image, &mut rgb)?;

        assert_eq!(rgb.num_channels(), 3);
        assert_eq!(rgb.size().width, 1);
        assert_eq!(rgb.size().height, 2);

        for (a, b) in rgb.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-1f32);
            // assert_eq!(a, b);
        }

        Ok(())
    }
}
