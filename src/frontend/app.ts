
/**
 * Main application logic
 */

import { getCurrentUser, logoutUser, User } from './auth.js';
import { loadSession, addMessage, clearSession, initializeSession, SessionMessage } from './session.js';

const API_URL = 'http://localhost:8000';

let currentUser: User | null = null;
let isLoading = false;

/**
 * Initialize the application
 */
async function init() {
    // Check if user is logged in
    currentUser = getCurrentUser();
    
    if (!currentUser) {
        window.location.href = 'login.html';
        return;
    }
    
    // Initialize session
    await initializeSession(currentUser.number);
    
    // Load and display existing messages
    await loadAndDisplayMessages();
    
    // Setup event listeners
    setupEventListeners();
    
    // Update UI
    updateUI();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Logout button
    const logoutBtn = document.getElementById('logoutBtn');
    logoutBtn?.addEventListener('click', () => {
        logoutUser();
    });
    
    // New session button
    const newSessionBtn = document.getElementById('newSessionBtn');
    newSessionBtn?.addEventListener('click', async () => {
        if (confirm('¿Estás seguro de que quieres iniciar una nueva sesión? Se perderá el historial actual.')) {
            await clearSession();
            await loadAndDisplayMessages();
        }
    });
    
    // Send button
    const sendBtn = document.getElementById('sendBtn');
    sendBtn?.addEventListener('click', handleSendMessage);
    
    // Input field - Enter key
    const messageInput = document.getElementById('messageInput') as HTMLInputElement;
    messageInput?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
}

/**
 * Update UI elements
 */
function updateUI() {
    if (currentUser) {
        const userNameEl = document.getElementById('userName');
        if (userNameEl) {
            userNameEl.textContent = currentUser.name;
        }
    }
}

/**
 * Load and display messages from session
 */
async function loadAndDisplayMessages() {
    const session = await loadSession(currentUser?.number);
    if (!session) return;
    
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) return;
    
    messagesContainer.innerHTML = '';
    
    for (const message of session.messages) {
        displayMessage(message);
    }
    
    scrollToBottom();
}

/**
 * Display a message in the chat
 */
function displayMessage(message: SessionMessage) {
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) return;
    
    if (message.type === 'request') {
        // User request - right side
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message message-user';
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${escapeHtml(message.content)}</p>
                <span class="message-time">${formatTime(message.timestamp)}</span>
            </div>
        `;
        messagesContainer.appendChild(messageDiv);
    } else if (message.type === 'report') {
        // Report - full width below request
        const reportDiv = document.createElement('div');
        reportDiv.className = 'message message-report';
        
        let reportHtml = '';
        if (message.report_data) {
            reportHtml = renderReport(message.report_data);
        } else {
            reportHtml = `<p>${escapeHtml(message.content)}</p>`;
        }
        
        reportDiv.innerHTML = `
            <div class="report-content">
                ${reportHtml}
                <div class="report-actions">
                    <button class="btn-download" onclick="downloadReport(${JSON.stringify(message.report_data).replace(/"/g, '&quot;')})">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        Descargar PDF
                    </button>
                </div>
                <span class="message-time">${formatTime(message.timestamp)}</span>
            </div>
        `;
        messagesContainer.appendChild(reportDiv);
    }
    
    scrollToBottom();
}

/**
 * Render report HTML
 */
function renderReport(report: any): string {
    if (!report) return '<p>No hay datos del reporte</p>';
    
    let html = `
        <div class="report-header">
            <h3>📰 Reporte Personalizado</h3>
            <p class="report-meta">
                Generado: ${new Date(report.generated_at).toLocaleString('es-ES')} | 
                Artículos: ${report.articles_in_report || 0} / ${report.total_articles || 0}
            </p>
        </div>
    `;
    
    if (report.articles && report.articles.length > 0) {
        report.articles.forEach((article: any, index: number) => {
            html += `
                <div class="article-card">
                    <div class="article-title">${index + 1}. ${escapeHtml(article.title || 'Sin título')}</div>
                    <div class="article-meta">
                        Sección: ${escapeHtml(article.section || 'N/A')} | 
                        Score: <span class="score">${article.score?.toFixed(3) || '0.000'}</span>
                    </div>
                    <div class="article-summary">
                        <strong>Resumen:</strong> ${escapeHtml(article.summary || 'Sin resumen')}
                    </div>
                    ${article.justification?.matching_categories?.length > 0 ? `
                        <div class="article-justification">
                            <strong>Categorías coincidentes:</strong>
                            ${article.justification.matching_categories.map((cat: string) => 
                                `<span class="category-tag">${escapeHtml(cat)}</span>`
                            ).join('')}
                        </div>
                    ` : ''}
                    ${article.url ? `
                        <div class="article-link">
                            <a href="${escapeHtml(article.url)}" target="_blank">Ver artículo completo →</a>
                        </div>
                    ` : ''}
                </div>
            `;
        });
    } else {
        html += '<p>No se encontraron artículos relevantes.</p>';
    }
    
    return html;
}

/**
 * Handle send message
 */
async function handleSendMessage() {
    const messageInput = document.getElementById('messageInput') as HTMLInputElement;
    const sendBtn = document.getElementById('sendBtn') as HTMLButtonElement;
    
    if (!messageInput || !sendBtn) return;
    
    const message = messageInput.value.trim();
    if (!message || isLoading) return;
    
    // Clear input
    messageInput.value = '';
    
    // Add user message to session
    await addMessage('request', message);
    // Reload session to display the message
    await loadAndDisplayMessages();
    
    // Show loading state
    isLoading = true;
    sendBtn.disabled = true;
    messageInput.disabled = true;
    
    // Show loading message
    const loadingMessage: SessionMessage = {
        type: 'report',
        content: 'Generando reporte...',
        timestamp: new Date().toISOString()
    };
    displayMessage(loadingMessage);
    
    try {
        // Get user profile text
        const profileText = currentUser?.profile_text || message;
        
        // Send request to API
        const response = await fetch(`${API_URL}/recommendations/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                profile_text: profileText,
                max_articles: 10
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }
        
        const data = await response.json();
        const report = data.report;
        
        // Remove loading message
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer && messagesContainer.lastElementChild) {
            messagesContainer.removeChild(messagesContainer.lastElementChild);
        }
        
        // Add report message to session
        await addMessage('report', 'Reporte generado exitosamente', report);
        // Reload session to display the report
        await loadAndDisplayMessages();
        
    } catch (error: any) {
        // Remove loading message
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer && messagesContainer.lastElementChild) {
            messagesContainer.removeChild(messagesContainer.lastElementChild);
        }
        
        // Add error message to session
        const errorContent = `Error: ${error.message || 'No se pudo generar el reporte'}`;
        await addMessage('report', errorContent);
        // Reload session to display the error
        await loadAndDisplayMessages();
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
}

/**
 * Download report as PDF
 */
(window as any).downloadReport = async function(reportData: any) {
    if (!reportData) {
        alert('No hay datos del reporte para descargar');
        return;
    }
    
    try {
        // Call API to generate PDF
        const response = await fetch(`${API_URL}/reports/generate-pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                report: reportData,
                user_name: currentUser?.name
            })
        });
        
        if (!response.ok) {
            throw new Error('Error al generar PDF');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `reporte_${currentUser?.name || 'usuario'}_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error: any) {
        alert(`Error al descargar PDF: ${error.message}`);
    }
};

/**
 * Utility functions
 */
function escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(timestamp: string): string {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
    const messagesContainer = document.getElementById('messagesContainer');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

