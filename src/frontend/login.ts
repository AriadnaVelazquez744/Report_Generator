/**
 * Login page logic
 */

import { loginUser, registerUser } from './auth.js';

const API_URL = 'http://localhost:8000';

// Toggle between login and register views
document.getElementById('showRegister')?.addEventListener('click', (e) => {
    e.preventDefault();
    (document.getElementById('loginView') as HTMLElement).style.display = 'none';
    (document.getElementById('registerView') as HTMLElement).style.display = 'block';
});

document.getElementById('showLogin')?.addEventListener('click', (e) => {
    e.preventDefault();
    (document.getElementById('registerView') as HTMLElement).style.display = 'none';
    (document.getElementById('loginView') as HTMLElement).style.display = 'block';
});

// Password toggle functionality
function setupPasswordToggle(inputId: string, toggleId: string, eyeId: string, eyeSlashId: string) {
    const input = document.getElementById(inputId) as HTMLInputElement;
    const toggle = document.getElementById(toggleId);
    const eye = document.getElementById(eyeId);
    const eyeSlash = document.getElementById(eyeSlashId);
    
    if (!input || !toggle || !eye || !eyeSlash) return;
    
    toggle.addEventListener('click', () => {
        if (input.type === 'password') {
            input.type = 'text';
            eye.style.display = 'none';
            eyeSlash.style.display = 'block';
        } else {
            input.type = 'password';
            eye.style.display = 'block';
            eyeSlash.style.display = 'none';
        }
    });
}

// Initialize password toggles when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setupPasswordToggle('loginPassword', 'loginPasswordToggle', 'loginPasswordEye', 'loginPasswordEyeSlash');
        setupPasswordToggle('registerPassword', 'registerPasswordToggle', 'registerPasswordEye', 'registerPasswordEyeSlash');
    });
} else {
    setupPasswordToggle('loginPassword', 'loginPasswordToggle', 'loginPasswordEye', 'loginPasswordEyeSlash');
    setupPasswordToggle('registerPassword', 'registerPasswordToggle', 'registerPasswordEye', 'registerPasswordEyeSlash');
}

// Handle login form
document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target as HTMLFormElement;
    const formData = new FormData(form);
    const name = formData.get('name') as string;
    const password = formData.get('password') as string;
    
    const errorMessage = document.getElementById('errorMessage') as HTMLElement;
    errorMessage.style.display = 'none';
    
    try {
        const user = await loginUser(name, password);
        
        if (user) {
            // Redirect to main app
            window.location.href = 'index.html';
        } else {
            errorMessage.textContent = 'Usuario o contraseña incorrectos';
            errorMessage.style.display = 'block';
        }
    } catch (error: any) {
        let errorText = 'Error al iniciar sesión';
        if (error instanceof Error) {
            errorText = error.message;
        } else if (typeof error === 'string') {
            errorText = error;
        } else if (error?.detail) {
            errorText = error.detail;
        } else if (error?.message) {
            errorText = error.message;
        }
        errorMessage.textContent = errorText;
        errorMessage.style.display = 'block';
    }
});

// Handle register form
document.getElementById('registerForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target as HTMLFormElement;
    const formData = new FormData(form);
    const password = formData.get('password') as string;
    const name = formData.get('name') as string;
    const interestsValue = formData.get('interests');
    const additionalInterests = interestsValue ? String(interestsValue).trim() : '';
    
    // Get selected categories
    const categoryCheckboxes = document.querySelectorAll('#categoryGrid input[type="checkbox"]:checked');
    const selectedCategories = Array.from(categoryCheckboxes).map(cb => (cb as HTMLInputElement).value);
    
    const errorMessage = document.getElementById('errorMessage') as HTMLElement;
    errorMessage.style.display = 'none';
    
    // Validation
    if (!password || !name) {
        errorMessage.textContent = 'Por favor completa todos los campos requeridos';
        errorMessage.style.display = 'block';
        return;
    }
    
    try {
        const user = await registerUser(password, name, selectedCategories, additionalInterests);
        
        if (user) {
            // Redirect to main app
            window.location.href = 'index.html';
        } else {
            errorMessage.textContent = 'Error al registrar usuario';
            errorMessage.style.display = 'block';
        }
    } catch (error: any) {
        let errorText = 'Error al registrar usuario';
        if (error instanceof Error) {
            errorText = error.message;
        } else if (typeof error === 'string') {
            errorText = error;
        } else if (error?.detail) {
            errorText = error.detail;
        } else if (error?.message) {
            errorText = error.message;
        }
        errorMessage.textContent = errorText;
        errorMessage.style.display = 'block';
    }
});




