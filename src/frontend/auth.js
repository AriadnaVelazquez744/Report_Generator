/**
 * Authentication utilities for user management
 */
const API_URL = 'http://localhost:8000';
/**
 * Login user via API
 */
export async function loginUser(name, password) {
    try {
        const response = await fetch(`${API_URL}/api/users/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, password })
        });
        if (!response.ok) {
            if (response.status === 401) {
                return null;
            }
            throw new Error('Error al iniciar sesión');
        }
        const data = await response.json();
        const user = data.user;
        // Store current user in sessionStorage
        sessionStorage.setItem('currentUser', JSON.stringify(user));
        return user;
    }
    catch (error) {
        console.error('Error logging in:', error);
        throw error;
    }
}
/**
 * Register new user and process taste selector
 */
export async function registerUser(password, name, selectedCategories, additionalInterests) {
    try {
        const response = await fetch(`${API_URL}/api/users/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                password,
                name,
                selected_categories: selectedCategories,
                additional_interests: additionalInterests
            })
        });
        if (!response.ok) {
            let errorDetail = 'Error al registrar usuario';
            try {
                const error = await response.json();
                errorDetail = error.detail || error.message || errorDetail;
            }
            catch {
                errorDetail = `Error ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorDetail);
        }
        const data = await response.json();
        const user = data.user;
        // Store current user in sessionStorage
        sessionStorage.setItem('currentUser', JSON.stringify(user));
        return user;
    }
    catch (error) {
        console.error('Error registering user:', error);
        throw error;
    }
}
/**
 * Get current logged-in user
 */
export function getCurrentUser() {
    const userStr = sessionStorage.getItem('currentUser');
    if (userStr) {
        try {
            return JSON.parse(userStr);
        }
        catch (e) {
            return null;
        }
    }
    return null;
}
/**
 * Logout user
 */
export function logoutUser() {
    sessionStorage.removeItem('currentUser');
    localStorage.removeItem('currentSession');
    window.location.href = 'login.html';
}
