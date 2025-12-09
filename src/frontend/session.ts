/**
 * Session management utilities
 */

export interface SessionMessage {
    type: 'request' | 'report';
    content: string;
    timestamp: string;
    report_data?: any;
}

export interface Session {
    user_id: string;
    messages: SessionMessage[];
    created_at: string;
    updated_at: string;
}

const API_URL = 'http://localhost:8000';

/**
 * Load current session from API
 */
export async function loadSession(userId?: string): Promise<Session | null> {
    if (!userId) {
        return null;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/session?user_id=${encodeURIComponent(userId)}`);
        if (!response.ok) {
            throw new Error('Failed to load session');
        }
        return await response.json();
    } catch (error) {
        console.error('Error loading session:', error);
        // Fallback to localStorage
        const localSession = localStorage.getItem('current_session');
        if (localSession) {
            try {
                return JSON.parse(localSession);
            } catch (e) {
                // Ignore parse errors
            }
        }
        return null;
    }
}

/**
 * Save session (not directly used, messages are added via API)
 */
export async function saveSession(session: Session): Promise<boolean> {
    // Session is saved via API when messages are added
    localStorage.setItem('current_session', JSON.stringify(session));
    return true;
}

/**
 * Add message to session via API
 */
export async function addMessage(
    type: 'request' | 'report',
    content: string,
    reportData?: any
): Promise<boolean> {
    // Get current user ID from sessionStorage
    const currentUserStr = sessionStorage.getItem('currentUser');
    if (!currentUserStr) {
        return false;
    }
    
    try {
        const user = JSON.parse(currentUserStr);
        const response = await fetch(`${API_URL}/api/session/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: user.number,
                type,
                content,
                report_data: reportData
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to add message');
        }
        
        // Also update local session
        const session = await loadSession(user.number);
        if (session) {
            localStorage.setItem('current_session', JSON.stringify(session));
        }
        
        return true;
    } catch (error) {
        console.error('Error adding message:', error);
        return false;
    }
}

/**
 * Clear session (for new session or logout)
 */
export async function clearSession(): Promise<boolean> {
    // Get current user ID from sessionStorage
    const currentUserStr = sessionStorage.getItem('currentUser');
    if (!currentUserStr) {
        return true;
    }
    
    try {
        const user = JSON.parse(currentUserStr);
        const response = await fetch(`${API_URL}/api/session/clear`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: user.number
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to clear session');
        }
        
        // Also clear local session
        localStorage.removeItem('current_session');
        return true;
    } catch (error) {
        console.error('Error clearing session:', error);
        return false;
    }
}

/**
 * Get all messages from session
 */
export async function getSessionMessages(): Promise<SessionMessage[]> {
    const session = await loadSession();
    return session?.messages || [];
}

/**
 * Initialize session for a user
 */
export async function initializeSession(userId: string): Promise<Session> {
    const session = await loadSession(userId);
    if (session) {
        return session;
    }
    
    // If no session, API will create one on first load
    const newSession: Session = {
        user_id: userId,
        messages: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
    };
    
    localStorage.setItem('current_session', JSON.stringify(newSession));
    return newSession;
}

