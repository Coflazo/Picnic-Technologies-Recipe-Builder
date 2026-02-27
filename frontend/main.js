const API_BASE = 'http://localhost:8000';

const ui = {
    input: document.getElementById('recipeInput'),
    tier: document.getElementById('tierSelect'),
    btn: document.getElementById('submitBtn'),
    loading: document.getElementById('loadingState'),
    results: document.getElementById('resultsSection'),
    list: document.getElementById('resultsList'),
    total: document.getElementById('summaryTotal')
};

// No default text per request

ui.btn.addEventListener('click', async () => {
    const text = ui.input.value.trim();
    if (text.length < 10) return;

    ui.btn.disabled = true;
    ui.btn.textContent = 'Generating...';

    ui.loading.classList.remove('hidden');
    ui.results.classList.add('hidden');

    try {
        const res = await fetch(`${API_BASE}/api/shopping-list`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                recipe_text: text,
                price_tier: ui.tier.value
            })
        });

        if (!res.ok) throw new Error('api threw an error');

        const data = await res.json();
        render(data);
    } catch (err) {
        console.error(err);
        ui.list.innerHTML = `<p style="color:red">something went wrong. see console.</p>`;
        ui.results.classList.remove('hidden');
    } finally {
        ui.btn.disabled = false;
        ui.btn.textContent = 'Generate Shopping List';
        ui.loading.classList.add('hidden');
    }
});

function render(data) {
    ui.list.innerHTML = '';

    // tell the user if we came up empty
    if (!data.items?.length) {
        ui.list.innerHTML = '<p>no items found.</p>';
    } else {
        data.items.forEach(item => {
            const el = document.createElement('div');
            el.className = 'result-card';

            // just dump the item info in a simple row
            el.innerHTML = `
                <div>
                    <div class="item-main">${item.article.Raw_Name}</div>
                    <div class="item-sub">${item.packs_needed}x pack · ${item.article.Brand}</div>
                </div>
                <div class="item-price">€${item.total_price.toFixed(2)}</div>
            `;
            ui.list.appendChild(el);
        });
    }

    ui.total.textContent = `€${data.total_cost.toFixed(2)}`;
    ui.results.classList.remove('hidden');
}

// Help Bubble Logic
const helpBubble = document.getElementById('helpBtn');
const helpModal = document.getElementById('helpModal');
const closeHelpBtn = document.getElementById('closeHelpBtn');

if (helpBubble && helpModal) {
    helpBubble.addEventListener('click', () => {
        helpModal.classList.toggle('hidden');
    });
}

closeHelpBtn.addEventListener('click', () => {
    helpModal.classList.add('hidden');
});
