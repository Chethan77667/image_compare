# 🖼️ Image Comparison Web Application

A secure web application that compares two images and provides a similarity score from 0 to 100. Built with Python Flask, OpenCV, and a beautiful responsive UI. Features access code verification and Excel export functionality.

## ✨ Features

- **🔐 Access Control**: Secure login with access codes
- **Drag & Drop Interface**: Easy image upload with drag-and-drop functionality
- **Real-time Preview**: See uploaded images immediately
- **Similarity Scoring**: Get a percentage score (0-100) showing how similar the images are
- **Modern UI**: Beautiful, responsive design that works on all devices
- **Backend Storage**: First image is stored on the server for comparison
- **Multiple Formats**: Supports PNG, JPG, JPEG, GIF, and BMP formats
- **📊 Excel Export**: Automatically save comparison results to Excel file
- **Session Management**: Secure user sessions with logout functionality

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

5. **Enter an access code**:
   - Demo codes: `12345`, `67890`, `ABCDE`, `FGHIJ`, `KLMNO`

## 📖 How to Use

### 1. **Login with Access Code**
   - Enter one of the valid access codes
   - Click "Enter" to access the comparison tool

### 2. **Upload Images**
   - **First Image (Stored)**: Upload the reference image (stored in backend)
   - **Second Image (Compare)**: Upload the image to compare against the stored image

### 3. **Compare Images**
   - Click the "Compare Images" button
   - Wait for the analysis to complete

### 4. **View Results**
   - See the similarity score (0-100%)
   - View the visual progress bar
   - Read the descriptive result text

### 5. **Download Results**
   - Click "Download Excel Results" to get the complete data
   - Excel file contains: Code, Date, Similarity Score, Image names, Session ID

## 🛠️ Technical Details

### Backend (Python Flask)
- **Authentication**: Session-based access control with code verification
- **Image Processing**: Uses OpenCV for image manipulation
- **Similarity Algorithm**: Structural Similarity Index (SSIM) from scikit-image
- **File Storage**: Secure file uploads with validation
- **Excel Export**: Automatic saving of results using pandas and openpyxl
- **API Endpoints**: RESTful API for upload, comparison, and download

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Drag & Drop**: Native HTML5 drag and drop API
- **Real-time Updates**: Dynamic UI updates without page refresh
- **Error Handling**: User-friendly error messages and validation
- **Session Management**: Secure logout functionality

### Image Comparison Algorithm
The application uses the **Structural Similarity Index (SSIM)** which:
- Compares structural information between images
- Provides scores from -1 to 1 (converted to 0-100%)
- Accounts for luminance, contrast, and structure
- Is more perceptually relevant than pixel-by-pixel comparison

## 📁 Project Structure

```
image-comparison-app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   ├── login.html        # Access code login page
│   └── comparison_tool.html # Main comparison interface
├── uploads/              # Directory for stored images (created automatically)
└── comparison_results.xlsx # Excel file with results (created automatically)
```

## 🔧 Configuration

### Access Codes
You can modify the valid codes in `app.py`:
```python
VALID_CODES = ['12345', '67890', 'ABCDE', 'FGHIJ', 'KLMNO']
```

### File Size Limits
- Maximum file size: 16MB per image
- Supported formats: PNG, JPG, JPEG, GIF, BMP

### Server Settings
- Host: 0.0.0.0 (accessible from any IP)
- Port: 5000
- Debug mode: Enabled (for development)

### Excel Export
The application automatically creates an Excel file (`comparison_results.xlsx`) with columns:
- **Code**: The access code used
- **Date**: Timestamp of comparison
- **Similarity_Score**: Percentage similarity (0-100)
- **Stored_Image**: Name of the reference image
- **Compared_Image**: Name of the compared image
- **Session_ID**: Unique session identifier

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**:
   - Change the port in `app.py` line 116
   - Or kill the process using port 5000

3. **Upload directory issues**:
   - The `uploads/` directory is created automatically
   - Ensure the application has write permissions

4. **Image comparison fails**:
   - Check that both images are valid image files
   - Ensure images are not corrupted
   - Try with smaller image files

5. **Excel export issues**:
   - Ensure pandas and openpyxl are installed
   - Check write permissions in the application directory

## 🔒 Security Notes

- **Access Control**: Only users with valid codes can access the tool
- **Session Security**: Secure session management with logout functionality
- **File Validation**: File uploads are validated for type and size
- **Filename Sanitization**: Prevents path traversal attacks
- **Image-only Uploads**: Only image files are accepted

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application!

---

**Enjoy comparing your images securely! 🎉**
