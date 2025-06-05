# FLUKE Project Website

This is the academic project website for "FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing".

## Files Structure

```
website/
├── index.html          # Main HTML page
├── style.css          # CSS styles and responsive design
├── script.js          # JavaScript for interactivity
└── website_README.md  # This file
```

## Features

- **Modern Academic Design**: Clean, professional layout following academic paper website conventions
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Interactive Navigation**: Smooth scrolling and active section highlighting
- **Results Tables**: Professional data presentation with hover effects
- **Resource Links**: Easy access to code, dataset, and paper
- **Citation Copy**: One-click citation copying functionality
- **Animations**: Subtle scroll-based animations for better user experience

## Customization Guide

### 1. Update Project Information

**Replace placeholder content in `index.html`:**

- **Title**: Update the project title in `<title>` and `<h1>` tags
- **Authors**: Replace "Your Name" and "Co-author" with actual author names
- **Affiliations**: Update institution information
- **Abstract**: Replace with your actual abstract
- **Results**: Update the results table with real data from your experiments

### 2. Update Links

**In the hero section and resources section:**

- **Paper Link**: Replace `href="#"` with actual paper URL (arXiv, conference, etc.)
- **GitHub Link**: Update `https://github.com/yourusername/fluke` with your repository URL
- **Dataset Link**: Add link to your dataset (Hugging Face, Google Drive, etc.)
- **Documentation**: Add link to your documentation

### 3. Add Real Images

**Replace placeholder framework image:**

- Create a framework diagram/architecture image
- Replace the placeholder URL in the framework section
- Recommended size: 800x400px or similar aspect ratio

### 4. Update Citation

**In both `index.html` and `script.js`:**

- Update author names
- Add correct venue/journal name
- Update publication year
- Add DOI if available

### 5. Color Customization

**In `style.css`, modify CSS variables:**

```css
:root {
    --primary-color: #2563eb;     /* Main theme color */
    --primary-dark: #1d4ed8;      /* Darker shade for hover */
    --text-primary: #1e293b;      /* Main text color */
    /* ... other variables ... */
}
```

## Deployment Options

### 1. GitHub Pages
1. Push your website files to a GitHub repository
2. Go to repository Settings > Pages
3. Select source branch (usually `main`)
4. Your site will be available at `https://yourusername.github.io/repository-name`

### 2. Netlify
1. Drag and drop your website folder to [Netlify Deploy](https://app.netlify.com/drop)
2. Or connect your GitHub repository for automatic deployments

### 3. Custom Domain
- Update links and metadata for your custom domain
- Add proper meta tags for SEO
- Consider adding Google Analytics

## Technical Details

- **No Build Process**: Pure HTML/CSS/JavaScript - no compilation needed
- **External Dependencies**: 
  - Font Awesome icons (CDN)
  - Inter font from Google Fonts (CDN)
- **Browser Support**: Modern browsers (Chrome, Firefox, Safari, Edge)
- **Performance**: Optimized for fast loading with minimal external resources

## SEO Optimization

Consider adding these meta tags to `<head>`:

```html
<meta name="description" content="Your project description">
<meta name="keywords" content="NLP, machine learning, linguistic testing">
<meta property="og:title" content="FLUKE: A Task-Agnostic Framework for Linguistic Capability Testing">
<meta property="og:description" content="Your project description">
<meta property="og:image" content="link-to-your-preview-image">
```

## Tips for Academic Websites

1. **Keep it Simple**: Focus on content, not flashy design
2. **Mobile-First**: Many visitors will view on mobile devices
3. **Fast Loading**: Academic audiences value quick access to information
4. **Clear Navigation**: Make it easy to find papers, code, and data
5. **Professional Tone**: Maintain academic credibility in all content

## License

This website template is free to use for academic projects. Please consider crediting this template if you find it useful. 