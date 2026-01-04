
/**
 * Main application logic
 */

import { getCurrentUser, logoutUser, User } from './auth.js';
import { loadSession, addMessage, clearSession, initializeSession, SessionMessage } from './session.js';

const API_URL = 'http://localhost:8000';

let currentUser: User | null = null;
let isLoading = false;
let loadingMessageEl: HTMLElement | null = null;

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
    if (!messagesContainer) return null;

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
                    ${message.report_data ?
                `<div class="report-buttons">
                            <button class="btn-download" onclick="downloadReport(${JSON.stringify(message.report_data).replace(/"/g, '&quot;')})">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="7 10 12 15 17 10"></polyline>
                                    <line x1="12" y1="15" x2="12" y2="3"></line>
                                </svg>
                                Descargar PDF
                            </button>
                            <button class="btn-view" onclick="viewReport(${JSON.stringify(message.report_data).replace(/"/g, '&quot;')})">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                    <circle cx="12" cy="12" r="3"></circle>
                                </svg>
                                Ver en Navegador
                            </button>
                        </div>` :
                ''}
                </div>
                <span class="message-time">${formatTime(message.timestamp)}</span>
            </div>
        `;
        messagesContainer.appendChild(reportDiv);
    }

    scrollToBottom();
    return messagesContainer.lastElementChild as HTMLElement | null;
}

/**
 * Render report HTML
 */
function renderReport(report: any): string {
    if (!report) return '<p>No hay datos del reporte</p>';

    let html = `
        <div class="report-header">
            <h3>📰 Reporte Personalizado</h3>
            <div class="report-meta">
                <div class="meta-line">Generado: ${new Date(report.generated_at).toLocaleString('es-ES')}</div>
                <div class="meta-line">Total de artículos en este reporte: ${report.articles_stats[1] || 0}</div>
                <div class="meta-line">Total de artículos relevantes encontrados: ${report.articles_stats[0] || 0}</div>
                <div class="meta-line">
                    Categorías de Interés: ${report.categories_of_interest.map((cat: string) =>
        `<span class="category-tag">${escapeHtml(cat)}</span>`
    ).join('')}
                </div>
            </div>
        </div>
    `;

    if (report.articles && report.articles.length > 0) {
        report.articles.forEach((article: any, index: number) => {
            html += `
                <div class="article-card">
                    <div class="article-title">${index + 1}. ${escapeHtml(article.title || 'Sin título')}</div>
                    <div class="article_date">
                        Sección: ${escapeHtml(article.section || 'N/A')} 
                    </div>
                    <div class="article_date">
                        Fecha: ${escapeHtml(article.date || 'Sin fecha')}
                    </div>
                    <div class="article-summary">
                        <strong>Resumen:</strong> ${escapeHtml(article.summary || 'Sin resumen')}
                    </div>
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
    console.log('[UI] send click -> mensaje añadido a la sesión');
    // Reload session to display the message
    await loadAndDisplayMessages();

    // Show loading state
    console.log('[UI] send click -> loading');
    setLoadingState(true);

    // Show loading message
    const loadingMessage: SessionMessage = {
        type: 'report',
        content: 'Generando reporte...',
        timestamp: new Date().toISOString()
    };
    loadingMessageEl = displayMessage(loadingMessage);

    try {
        console.log('[API] start generate-text-report');
        const t0 = performance.now();
        console.log('[API] start generate-from-input - sending request');
        // Send request to API (endpoint con estructura de respuesta correcta)
        const response = await fetch(`${API_URL}/recommendations/generate-text-report`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: currentUser?.number || currentUser?.name,
                user_input: message,
                max_articles: 10,
                input_weight: 0.7,
                prioritize_recent: true
            })
        });

        console.log('[API] generate-text-report ended successfully');

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('[API] done', { elapsed_ms: Math.round(performance.now() - t0), status: data.status });

        removeLoadingMessage();

        // Add structured report message to session
        const reportDataWithQuery = {
            ...data.structured_report,
            user_query: message  // Guardar la query original del usuario
        };
        await addMessage('report', 'Reporte generado exitosamente', reportDataWithQuery);
        // Reload session to display the report
        await loadAndDisplayMessages();

    } catch (error: any) {
        console.error('[API] error', error);
        removeLoadingMessage();

        // Add error message to session
        const errorContent = `Error: ${error.message || 'No se pudo generar el reporte'}`;
        await addMessage('report', errorContent);
        // Reload session to display the error
        await loadAndDisplayMessages();
    } finally {
        setLoadingState(false);
        messageInput.focus();
    }
}

/**
 * Download report as PDF
 */
(window as any).downloadReport = async function (reportData: any) {
    if (!reportData) {
        alert('No hay datos del reporte para descargar');
        return;
    }

    // Pedir ubicación personalizada
    const customPath = await showPathDialog();

    try {
        // Call API to generate PDF
        const response = await fetch(`${API_URL}/reports/generate-pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                report: reportData,
                user_name: currentUser?.name,
                user_query: reportData.user_query || '',
                custom_path: customPath || null,
                browser_mode: false
            })
        });

        if (!response.ok) {
            throw new Error('Error al generar PDF');
        }

        const result = await response.json();

        if (result.success) {
            // Descargar el PDF generado
            const downloadResponse = await fetch(`${API_URL}/reports/download-pdf?filename=${result.filename}`);
            const blob = await downloadResponse.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = result.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            // Mostrar información del archivo
            alert(`PDF generado exitosamente:\nArchivo: ${result.filename}\nTamaño: ${(result.size / 1024).toFixed(2)} KB`);
        }

    } catch (error: any) {
        alert(`Error al descargar PDF: ${error.message}`);
    }
};

/**
 * View report in browser
 */
(window as any).viewReport = async function (reportData: any) {
    if (!reportData) {
        alert('No hay datos del reporte para visualizar');
        return;
    }

    try {
        // Call API to generate PDF for browser viewing
        const response = await fetch(`${API_URL}/reports/generate-pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                report: reportData,
                user_name: currentUser?.name,
                user_query: reportData.user_query || '',
                browser_mode: true
            })
        });

        if (!response.ok) {
            throw new Error('Error al generar PDF para visualización');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        // Abrir PDF en nueva pestaña
        window.open(url, '_blank');

        // Limpiar URL después de un tiempo
        setTimeout(() => {
            window.URL.revokeObjectURL(url);
        }, 60000);

    } catch (error: any) {
        alert(`Error al visualizar PDF: ${error.message}`);
    }
};

/**
 * Show dialog for custom path input
 */
async function showPathDialog(): Promise<string | null> {
    return new Promise((resolve) => {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;

        const dialog = document.createElement('div');
        dialog.style.cssText = `
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            width: 90%;
        `;

        dialog.innerHTML = `
            <h3 style="margin: 0 0 15px 0; color: #333;">Configurar Ubicación del PDF</h3>
            <p style="margin: 0 0 15px 0; color: #666;">Deje en blanco para usar la ubicación por defecto o ingrese una ruta personalizada:</p>
            <input type="text" id="customPathInput" placeholder="Ej: /home/user/mis_reportes/reporte_personalizado.pdf" 
                   style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 15px; box-sizing: border-box;">
            <div style="text-align: right;">
                <button id="cancelBtn" style="margin-right: 10px; padding: 8px 16px; border: 1px solid #ddd; background: #f5f5f5; border-radius: 4px; cursor: pointer;">Cancelar</button>
                <button id="acceptBtn" style="padding: 8px 16px; border: none; background: #007bff; color: white; border-radius: 4px; cursor: pointer;">Aceptar</button>
            </div>
        `;

        modal.appendChild(dialog);
        document.body.appendChild(modal);

        const input = dialog.querySelector('#customPathInput') as HTMLInputElement;
        const cancelBtn = dialog.querySelector('#cancelBtn') as HTMLButtonElement;
        const acceptBtn = dialog.querySelector('#acceptBtn') as HTMLButtonElement;

        const cleanup = () => {
            document.body.removeChild(modal);
        };

        const handleAccept = () => {
            const path = input.value.trim() || null;
            cleanup();
            resolve(path);
        };

        const handleCancel = () => {
            cleanup();
            resolve(null);
        };

        acceptBtn.addEventListener('click', handleAccept);
        cancelBtn.addEventListener('click', handleCancel);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleAccept();
            }
        });

        // Focus en el input
        setTimeout(() => input.focus(), 100);
    });
}

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

function setLoadingState(loading: boolean) {
    isLoading = loading;
    const sendBtn = document.getElementById('sendBtn') as HTMLButtonElement | null;
    const messageInput = document.getElementById('messageInput') as HTMLInputElement | null;
    if (!sendBtn || !messageInput) return;

    sendBtn.disabled = loading;
    // messageInput.disabled = loading;

    if (loading) {
        sendBtn.classList.add('loading');
        sendBtn.innerHTML = `
            <svg class="spinner" viewBox="0 0 50 50">
                <circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle>
            </svg>
        `;
    } else {
        sendBtn.classList.remove('loading');
        sendBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
        `;
    }
}

function removeLoadingMessage() {
    if (!loadingMessageEl) return;
    const messagesContainer = document.getElementById('messagesContainer');
    if (messagesContainer && messagesContainer.contains(loadingMessageEl)) {
        messagesContainer.removeChild(loadingMessageEl);
    }
    loadingMessageEl = null;
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

export { };
