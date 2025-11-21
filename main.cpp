#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QTextStream>
#include <QString>

#include <vector>
#include <complex>
#include <cmath>

#include <fftw3.h>

using namespace std;

const QString file_name = "gray_lenna_small.png";

// ----------- FFT 2D - calculates images DFT -----------
void fft2d(
        const QImage &image,
        vector< vector< complex<double> > > &trs
        )
{
    int m = image.height();
    int n = image.width();

    trs.assign(m, vector< complex<double> >(n));

    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m * n);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m * n);

    for (int r = 0; r < m; r++)
    {
        const quint8* row_ptr = (const quint8*)(image.bits() + r * image.bytesPerLine());
        for (int c = 0; c < n; c++)
        {
            double centered = row_ptr[c] * pow(-1.0, r + c);
            in[r*n + c][0] = centered; // real
            in[r*n + c][1] = 0.0;      // imag
        }
    }

    fftw_plan plan = fftw_plan_dft_2d(
            m, n, in, out,
            FFTW_FORWARD, FFTW_ESTIMATE
        );

    fftw_execute(plan);

    // convert output from FFTW into std::complex<double>
    for (int r = 0; r < m; r++)
    {
        for (int c = 0; c < n; c++)
        {
            //out is liner array, so index is r*n + c
            trs[r][c] = complex<double>(out[r*n + c][0], out[r*n + c][1]);
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

// ----------- Low-pass filter -----------
void lowPassFilter(
        vector< vector< complex<double> > > &trs,
        double keep_ratio   // fraction of low frequencies to keep (0.0–1.0)
        )
{
    int m = trs.size();
    int n = trs[0].size();

    int half_m = m / 2;
    int half_n = n / 2;

    int keep_m = (int)(keep_ratio * half_m);
    int keep_n = (int)(keep_ratio * half_n);

    for (int r = 0; r < m; r++)
    {
        for (int c = 0; c < n; c++)
        {
            bool outside_row = (r < half_m - keep_m) || (r > half_m + keep_m);
            bool outside_col = (c < half_n - keep_n) || (c > half_n + keep_n);
            if (outside_row || outside_col)
                trs[r][c] = 0.0;
        }
    }
}

// ----------- Inverse FFT -----------
void inverseFFT(
        const vector< vector< complex<double> > > &trs,
        QImage &result_img
        )
{
    int m = result_img.height();
    int n = result_img.width();

    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m * n);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m * n);

    for (int r = 0; r < m; r++)
    {
        for (int c = 0; c < n; c++)
        {
            in[r*n + c][0] = trs[r][c].real();
            in[r*n + c][1] = trs[r][c].imag();
        }
    }

    fftw_plan plan = fftw_plan_dft_2d(
            m, n, in, out,
            FFTW_BACKWARD, FFTW_ESTIMATE
        );

    fftw_execute(plan);

    for (int r = 0; r < m; r++)
    {
        quint8* ptr_row = (quint8*)(result_img.bits() + r * result_img.bytesPerLine());
        for (int c = 0; c < n; c++)
        {
            double value = out[r*n + c][0] / (m * n); // normalize
            value *= pow(-1.0, r + c); // undo centering
            int pixel = std::round(value);
            if (pixel < 0) pixel = 0;
            if (pixel > 255) pixel = 255;
            ptr_row[c] = pixel;
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

// ----------- Main -----------
int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QImage image;
    QLabel label;

    if (image.load(file_name))
    {
        QTextStream(stdout) << "Image loaded: " << file_name << Qt::endl;
        QTextStream(stdout) << "Format: " << image.format() << Qt::endl;

        if (image.format() == QImage::Format_Grayscale8)
        {
            vector< vector< complex<double> > > trs;

            // FFT
            fft2d(image, trs);

            // Low-pass filter: keep 10% of low frequencies
            lowPassFilter(trs, 0.10);

            // Inverse FFT
            QImage restored(image.height(), image.width(), QImage::Format_Grayscale8);
            inverseFFT(trs, restored);

            // show results
            label.setPixmap(QPixmap::fromImage(restored));
        }
    }
    else
    {
        QTextStream(stdout) << "Cannot load image: " << file_name << Qt::endl;
    }

    label.show();
    return app.exec();
}
