/**
 * Authentication utilities for user management
 */

const API_URL = 'http://localhost:8000';

export interface User {
    number: string;
    password: string;
    name: string;
    profile_text: string;
    categories: string[];
    entities: Array<{ text: string; label: string }>;
    vector: number[];
    created_at: string;
    updated_at: string;
}

/**
 * Login user via API
 */
export async function loginUser(name: string, password: string): Promise<User | null> {
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
    } catch (error) {
        console.error('Error logging in:', error);
        throw error;
    }
}

/**
 * Register new user and process taste selector
 */
export async function registerUser(
    password: string,
    name: string,
    selectedCategories: string[],
    additionalInterests: string
): Promise<User | null> {
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
            } catch {
                errorDetail = `Error ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorDetail);
        }
        
        const data = await response.json();
        const user = data.user;
        
        // Store current user in sessionStorage
        sessionStorage.setItem('currentUser', JSON.stringify(user));
        return user;
    } catch (error: any) {
        console.error('Error registering user:', error);
        throw error;
    }
}

/**
 * Get current logged-in user
 */
export function getCurrentUser(): User | null {
    const userStr = sessionStorage.getItem('currentUser');
    if (userStr) {
        try {
            return JSON.parse(userStr);
        } catch (e) {
            return null;
        }
    }
    return null;
}

/**
 * Logout user
 */
export function logoutUser(): void {
    sessionStorage.removeItem('currentUser');
    localStorage.removeItem('currentSession');
    window.location.href = 'login.html';
}

