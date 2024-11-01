# 🎯 SVD Digital Watermarking System

## 📝 Description
A robust digital watermarking system that uses Singular Value Decomposition (SVD) to embed invisible watermarks into images. Features batch processing, noise reduction, and parallel execution for optimal performance.

## ✨ Key Features
• 🔒 Invisible watermark embedding
• 🔍 Robust watermark recovery
• 🎨 Advanced noise reduction
• ⚡ Batch processing optimization
• 📊 Real-time progress monitoring
• ⚙️ Highly customizable parameters
• 🚀 Parallel processing support

## 📋 Requirements
• Python 3.8+
• numpy >= 1.19.0
• opencv-python >= 4.5.0
• numba >= 0.53.0

## 💻 Quick Start
1. Install dependencies:
   pip install numpy opencv-python numba

2. Basic usage:
   python watermark_SVD.py --image "path/to/image.png" --watermark "path/to/logo.png" --output "path/to/output.png"

3. Advanced usage:
   python watermark_SVD.py --image "path/to/image.png" --watermark "path/to/logo.png" --alpha 0.0001 --noise-reduction 0.6 --batch-size 64 --output "path/to/marked.png" --recovered-output "path/to/recovered.png"

## 🔧 Parameters Explained
Required:
• --image: Path to the input image
• --watermark: Path to the watermark image (preferably black and white)

Optional:
• --alpha: Watermark strength (default: 0.0001)
  - Lower values: More invisible but less robust
  - Higher values: More robust but might be visible
  - Recommended range: 0.00001 - 0.001

• --noise-reduction: Noise reduction level (default: 0.7)
  - Range: 0.0 - 1.0
  - Lower values: More detail retention
  - Higher values: More noise reduction

• --batch-size: Processing batch size (default: 64)
  - Affects processing speed and memory usage
  - Higher values: Faster but more memory intensive

• --output: Path for watermarked image
• --recovered-output: Path for recovered watermark

## 🔬 What is SVD?
Singular Value Decomposition (SVD) is a mathematical technique that decomposes a matrix A into three matrices:
A = U × Σ × V^T

Where:
• U and V are orthogonal matrices
• Σ contains singular values in descending order

In watermarking:
1. The image is divided into blocks
2. SVD is applied to each block
3. Singular values are modified to embed the watermark
4. The modified blocks are reconstructed

Benefits:
• Stability of singular values
• Resistance to common image modifications
• Preservation of image quality

## 🎯 Use Cases and Applications
1. Copyright Protection
   • Prove content ownership
   • Track unauthorized usage
   • Legal proof of ownership

2. Content Authentication
   • Verify content integrity
   • Detect tampering
   • Ensure authenticity

3. Document Security
   • Secure confidential documents
   • Add hidden verification marks
   • Track document distribution

4. Digital Art Protection
   • Protect digital artwork
   • Prove authorship
   • Track licensed usage

## ⚙️ Best Practices
Image Selection:
• Use high-quality source images
• Ensure 4:1 size ratio (image:watermark)
• Prefer uncompressed formats

Watermark Design:
• Use simple, clear designs
• Black and white logos work best
• Avoid complex patterns

Parameter Selection:
• Start with default values
• Test with sample images first
• Adjust based on requirements

## 🚀 Performance Tips
For Speed:
• Increase batch size
• Use smaller watermarks
• Process grayscale images

For Quality:
• Use lower alpha values
• Adjust noise reduction
• Use clear watermark images

For Memory:
• Reduce batch size
• Process smaller images
• Clear unused variables

## ⚠️ Limitations
• Input image must be 4x larger than watermark
• Works best with grayscale images
• Color images are converted to grayscale
• Memory usage scales with batch size
• Processing time depends on image size

## 📊 Examples
Good watermark characteristics:
• Simple logos
• Clear text
• High contrast designs
• Binary (black and white) images

Bad watermark characteristics:
• Complex patterns
• Gradients
• Fine details
• Low contrast designs

## 📝 License
This project is licensed under the MIT License.
