/**
 * BODYBALANCE.AI Embeddable Chat Widget
 * 
 * Usage:
 * <script src="https://your-domain.com/widget.js"></script>
 * <script>
 *   BodyBalanceWidget.init({
 *     apiUrl: 'https://your-api-domain.com',
 *     position: 'bottom-right',  // or 'bottom-left'
 *     primaryColor: '#2E7D32',
 *     title: 'BODYBALANCE.AI',
 *     greeting: 'Hi! How can I help you today?'
 *   });
 * </script>
 */

(function(window) {
    'use strict';

    const BodyBalanceWidget = {
        config: {
            apiUrl: '',
            position: 'bottom-right',
            primaryColor: '#2E7D32',
            title: 'BODYBALANCE.AI',
            greeting: '👋 Hi! How can I help you today?'
        },
        sessionId: null,
        isOpen: false,
        container: null,

        init: function(options) {
            // Merge options with defaults
            this.config = { ...this.config, ...options };
            
            if (!this.config.apiUrl) {
                console.error('BodyBalanceWidget: apiUrl is required');
                return;
            }

            this.createWidget();
            this.attachEventListeners();
        },

        createWidget: function() {
            const position = this.config.position === 'bottom-left' 
                ? 'left: 20px;' 
                : 'right: 20px;';

            const widgetHTML = `
                <div id="bb-widget-container" style="
                    position: fixed;
                    bottom: 20px;
                    ${position}
                    z-index: 9999;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                ">
                    <!-- Chat Window -->
                    <div id="bb-chat-window" style="
                        display: none;
                        width: 370px;
                        height: 520px;
                        background: white;
                        border-radius: 16px;
                        box-shadow: 0 5px 40px rgba(0,0,0,0.16);
                        flex-direction: column;
                        overflow: hidden;
                        margin-bottom: 16px;
                    ">
                        <!-- Header -->
                        <div id="bb-header" style="
                            background: ${this.config.primaryColor};
                            color: white;
                            padding: 16px 20px;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        ">
                            <div>
                                <div style="font-weight: 600; font-size: 16px;">💪 ${this.config.title}</div>
                                <div style="font-size: 12px; opacity: 0.9;">Online | Typically replies instantly</div>
                            </div>
                            <button id="bb-close" style="
                                background: none;
                                border: none;
                                color: white;
                                font-size: 20px;
                                cursor: pointer;
                                padding: 0;
                                line-height: 1;
                            ">✕</button>
                        </div>

                        <!-- Messages -->
                        <div id="bb-messages" style="
                            flex: 1;
                            overflow-y: auto;
                            padding: 16px;
                            background: #f8f9fa;
                        ">
                            <div class="bb-message bb-bot" style="
                                background: white;
                                padding: 12px 16px;
                                border-radius: 16px;
                                border-bottom-left-radius: 4px;
                                margin-bottom: 12px;
                                max-width: 85%;
                                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                            ">${this.config.greeting}</div>
                        </div>

                        <!-- Input -->
                        <div style="
                            padding: 12px 16px;
                            border-top: 1px solid #eee;
                            background: white;
                            display: flex;
                            gap: 8px;
                        ">
                            <input type="text" id="bb-input" placeholder="Type your message..." style="
                                flex: 1;
                                padding: 12px 16px;
                                border: 1px solid #e0e0e0;
                                border-radius: 24px;
                                outline: none;
                                font-size: 14px;
                                transition: border-color 0.2s;
                            ">
                            <button id="bb-send" style="
                                width: 44px;
                                height: 44px;
                                background: ${this.config.primaryColor};
                                color: white;
                                border: none;
                                border-radius: 50%;
                                cursor: pointer;
                                font-size: 18px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                transition: transform 0.2s;
                            ">➤</button>
                        </div>

                        <!-- Powered by -->
                        <div style="
                            text-align: center;
                            padding: 8px;
                            font-size: 11px;
                            color: #999;
                            background: #f8f9fa;
                        ">
                            Powered by <a href="#" style="color: ${this.config.primaryColor}; text-decoration: none;">BODYBALANCE.AI</a>
                        </div>
                    </div>

                    <!-- Toggle Button -->
                    <button id="bb-toggle" style="
                        width: 60px;
                        height: 60px;
                        border-radius: 50%;
                        background: ${this.config.primaryColor};
                        color: white;
                        border: none;
                        cursor: pointer;
                        font-size: 28px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        transition: transform 0.2s, box-shadow 0.2s;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">💬</button>
                </div>
            `;

            // Create container and add to body
            const div = document.createElement('div');
            div.innerHTML = widgetHTML;
            document.body.appendChild(div);

            this.container = document.getElementById('bb-widget-container');
        },

        attachEventListeners: function() {
            const self = this;

            // Toggle button
            document.getElementById('bb-toggle').addEventListener('click', function() {
                self.toggle();
            });

            // Close button
            document.getElementById('bb-close').addEventListener('click', function() {
                self.toggle();
            });

            // Send button
            document.getElementById('bb-send').addEventListener('click', function() {
                self.sendMessage();
            });

            // Enter key
            document.getElementById('bb-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    self.sendMessage();
                }
            });

            // Input focus effect
            const input = document.getElementById('bb-input');
            input.addEventListener('focus', function() {
                this.style.borderColor = self.config.primaryColor;
            });
            input.addEventListener('blur', function() {
                this.style.borderColor = '#e0e0e0';
            });

            // Button hover effects
            const sendBtn = document.getElementById('bb-send');
            sendBtn.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.05)';
            });
            sendBtn.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });

            const toggleBtn = document.getElementById('bb-toggle');
            toggleBtn.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.1)';
                this.style.boxShadow = '0 6px 20px rgba(0,0,0,0.2)';
            });
            toggleBtn.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
                this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            });
        },

        toggle: function() {
            this.isOpen = !this.isOpen;
            const chatWindow = document.getElementById('bb-chat-window');
            const toggleBtn = document.getElementById('bb-toggle');

            if (this.isOpen) {
                chatWindow.style.display = 'flex';
                toggleBtn.style.display = 'none';
                document.getElementById('bb-input').focus();
            } else {
                chatWindow.style.display = 'none';
                toggleBtn.style.display = 'flex';
            }
        },

        addMessage: function(text, isUser) {
            const messagesDiv = document.getElementById('bb-messages');
            const messageDiv = document.createElement('div');
            
            messageDiv.style.cssText = `
                padding: 12px 16px;
                border-radius: 16px;
                margin-bottom: 12px;
                max-width: 85%;
                word-wrap: break-word;
                ${isUser ? `
                    background: ${this.config.primaryColor};
                    color: white;
                    margin-left: auto;
                    border-bottom-right-radius: 4px;
                ` : `
                    background: white;
                    border-bottom-left-radius: 4px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                `}
            `;
            
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        },

        addTypingIndicator: function() {
            const messagesDiv = document.getElementById('bb-messages');
            const indicator = document.createElement('div');
            indicator.id = 'bb-typing';
            indicator.style.cssText = `
                background: white;
                padding: 12px 16px;
                border-radius: 16px;
                border-bottom-left-radius: 4px;
                margin-bottom: 12px;
                max-width: 85%;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            `;
            indicator.innerHTML = '<span style="animation: blink 1s infinite;">●</span> <span style="animation: blink 1s infinite 0.2s;">●</span> <span style="animation: blink 1s infinite 0.4s;">●</span>';
            messagesDiv.appendChild(indicator);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Add blink animation
            if (!document.getElementById('bb-styles')) {
                const style = document.createElement('style');
                style.id = 'bb-styles';
                style.textContent = '@keyframes blink { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }';
                document.head.appendChild(style);
            }
        },

        removeTypingIndicator: function() {
            const indicator = document.getElementById('bb-typing');
            if (indicator) {
                indicator.remove();
            }
        },

        async sendMessage: function() {
            const input = document.getElementById('bb-input');
            const message = input.value.trim();
            
            if (!message) return;

            // Add user message
            this.addMessage(message, true);
            input.value = '';

            // Show typing indicator
            this.addTypingIndicator();

            try {
                const response = await fetch(this.config.apiUrl + '/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: this.sessionId
                    })
                });

                const data = await response.json();
                this.sessionId = data.session_id;

                // Remove typing indicator and add response
                this.removeTypingIndicator();
                this.addMessage(data.response, false);

                // Show suggestions if available
                if (data.suggestions && data.suggestions.length > 0) {
                    setTimeout(() => {
                        this.addMessage('You might also want to ask about:\n• ' + data.suggestions.join('\n• '), false);
                    }, 500);
                }

            } catch (error) {
                console.error('BodyBalanceWidget error:', error);
                this.removeTypingIndicator();
                this.addMessage('Sorry, something went wrong. Please try again.', false);
            }
        }
    };

    // Expose to global scope
    window.BodyBalanceWidget = BodyBalanceWidget;

})(window);
