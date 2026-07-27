/**
 * custom_select.js - 自定义下拉框 (替代原生 select, 解决 Windows 白底问题)
 * 
 * 自动替换 #sidebar 内的所有 <select> 为可完全自定义样式的下拉组件。
 */

class CustomSelect {
    /**
     * @param {HTMLSelectElement} selectEl - 原生 select 元素
     */
    constructor(selectEl) {
        this.select = selectEl;
        this.options = [];
        this._open = false;

        // 隐藏原生 select
        this.select.classList.add('custom-select-hidden');

        // 监听原生 select 的变化 (当外部 JS 修改 options 时自动同步)
        this._observer = new MutationObserver(() => {
            this._buildOptions();
            this._updateValue();
        });
        this._observer.observe(this.select, {
            childList: true,
            subtree: true,
            characterData: true,
        });

        // 构建自定义容器
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'custom-select';

        // 触发器 (显示当前选中项)
        this.trigger = document.createElement('div');
        this.trigger.className = 'select-trigger';
        this.trigger.innerHTML = `
            <span class="select-value"></span>
            <span class="arrow"></span>
        `;
        this.valueEl = this.trigger.querySelector('.select-value');

        // 选项容器
        this.optionsContainer = document.createElement('div');
        this.optionsContainer.className = 'select-options';

        // 插入到 DOM
        this.select.parentNode.insertBefore(this.wrapper, this.select);
        this.wrapper.appendChild(this.trigger);
        this.wrapper.appendChild(this.optionsContainer);
        this.wrapper.appendChild(this.select);

        // 构建选项
        this._buildOptions();

        // 更新显示
        this._updateValue();

        // 绑定事件
        this._bindEvents();
    }

    _buildOptions() {
        this.optionsContainer.innerHTML = '';
        this.options = [];

        for (const opt of this.select.options) {
            const item = document.createElement('div');
            item.className = 'select-option';
            if (opt.selected) item.classList.add('selected');
            item.textContent = opt.text;
            item.dataset.value = opt.value;

            item.addEventListener('click', (e) => {
                e.stopPropagation();
                this._select(opt.value, opt.text);
                this._close();
            });

            this.optionsContainer.appendChild(item);
            this.options.push(item);
        }
    }

    _select(value, text) {
        this.select.value = value;
        // 触发 change 事件
        this.select.dispatchEvent(new Event('change', { bubbles: true }));
        this._updateValue();
    }

    _updateValue() {
        const selectedOpt = this.select.options[this.select.selectedIndex];
        if (selectedOpt) {
            this.valueEl.textContent = selectedOpt.text;
        }

        // 更新选项高亮
        for (const item of this.options) {
            item.classList.toggle('selected', item.dataset.value === this.select.value);
        }
    }

    _toggle() {
        if (this._open) {
            this._close();
        } else {
            this._openDropdown();
        }
    }

    _openDropdown() {
        // 关闭其他所有自定义下拉框
        document.querySelectorAll('.custom-select.open').forEach(el => {
            el.classList.remove('open');
        });

        this.wrapper.classList.add('open');
        this._open = true;

        // 滚动到选中项
        const selected = this.optionsContainer.querySelector('.select-option.selected');
        if (selected) {
            selected.scrollIntoView({ block: 'nearest' });
        }
    }

    _close() {
        this.wrapper.classList.remove('open');
        this._open = false;
    }

    _bindEvents() {
        // 点击触发器切换
        this.trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            this._toggle();
        });

        // 点击外部关闭
        document.addEventListener('click', (e) => {
            if (!this.wrapper.contains(e.target)) {
                this._close();
            }
        });

        // 监听原生 select 的变化 (来自外部)
        this.select.addEventListener('change', () => {
            this._updateValue();
        });

        // 键盘支持
        this.trigger.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this._toggle();
            }
            if (e.key === 'Escape') {
                this._close();
            }
        });
        this.trigger.setAttribute('tabindex', '0');
        this.trigger.setAttribute('role', 'combobox');
    }

    /** 刷新选项列表 (当 select 内容变化时调用) */
    refresh() {
        this._buildOptions();
        this._updateValue();
    }

    /** 销毁 */
    destroy() {
        if (this._observer) {
            this._observer.disconnect();
            this._observer = null;
        }
        this.select.classList.remove('custom-select-hidden');
        this.wrapper.parentNode.removeChild(this.wrapper);
    }
}

/**
 * 初始化侧边栏所有自定义下拉框
 */
function initCustomSelects() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    const selects = sidebar.querySelectorAll('select');
    selects.forEach(sel => {
        // 跳过已经转换的
        if (sel.classList.contains('custom-select-hidden')) return;
        new CustomSelect(sel);
    });
}

// 在 DOM 加载完成后自动初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCustomSelects);
} else {
    initCustomSelects();
}