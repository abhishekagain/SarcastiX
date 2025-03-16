mkdir : A positional parameter cannot be found that accepts argument 
'C:\Users\LENOVO\Desktop\ProjectHackDU\static'.
At line:1 char:1
+ mkdir "C:\Users\LENOVO\Desktop\ProjectHackDU" "C:\Users\LENOVO\Deskto ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) [mkdir], ParameterBindingException
    + FullyQualifiedErrorId : PositionalParameterNotFound,mkdir
 
PS C:\Users\LENOVO\Desktop># UxInspiredDesign

Developed by Abhisek Yadav (Team Leader - Code Carrots)

## File Structure
```
UxInspiredDesign/
├── src/
│   ├── components/
│   │   ├── Header/
│   │   │   ├── Header.jsx
│   │   │   └── Header.css
│   │   ├── Footer/
│   │   │   ├── Footer.jsx
│   │   │   └── Footer.css
│   │   └── UI/
│   │       ├── Button.jsx
│   │       └── Card.jsx
│   ├── pages/
│   │   ├── Home/
│   │   │   ├── Home.jsx
│   │   │   └── Home.css
│   │   ├── Dashboard/
│   │   │   ├── Dashboard.jsx
│   │   │   └── Dashboard.css
│   │   └── Auth/
│   │       ├── Login.jsx
│   │       └── Register.jsx
│   ├── utils/
│   │   ├── api.js
│   │   └── helpers.js
│   └── styles/
│       ├── global.css
│       └── variables.css
├── public/
│   ├── assets/
│   │   ├── images/
│   │   └── icons/
│   └── index.html
├── config/
│   └── config.js
├── .env
├── .gitignore
├── package.json
└── README.md
```

## Quick Start
- **Online (Replit)**: Click the "Run" button
- **Offline**: Follow setup instructions below

## Setup Instructions
1. Install dependencies:
   ```bash
   npm install
   ```

2. Create `.env` file:
   ```
   REACT_APP_API_URL=your_api_url
   REACT_APP_SECRET_KEY=your_secret_key
   ```

3. Start development server:
   ```bash
   npm start
   ```

## Default Credentials
- Username: `admin`
- Password: `admin`

## Features
- Modern UI/UX Design
- Responsive Layout
- Component-Based Architecture
- Custom UI Components
- Theme Support

## Tech Stack
- React.js
- CSS3/SASS
- React Router
- Context API
- Modern JavaScript (ES6+)

## Important Notes
- Use consistent naming conventions
- Follow component structure
- Keep styles modular
- Document components 