class BeforeAfter {
    constructor(enteryObject) {

        const beforeAfterContainer = document.querySelector(enteryObject.id);
        const before = beforeAfterContainer.querySelector('.bal-before');
        const beforeText = beforeAfterContainer.querySelector('.bal-beforePosition');
        const afterText = beforeAfterContainer.querySelector('.bal-afterPosition');
        const handle = beforeAfterContainer.querySelector('.bal-handle');
        var widthChange = 0;

        beforeAfterContainer.querySelector('.bal-before-inset').setAttribute("style", "width: " + beforeAfterContainer.offsetWidth + "px;")
        window.onresize = function () {
            beforeAfterContainer.querySelector('.bal-before-inset').setAttribute("style", "width: " + beforeAfterContainer.offsetWidth + "px;")
        }
        before.setAttribute('style', "width: 50%;");
        handle.setAttribute('style', "left: 50%;");

        //touch screen event listener
        beforeAfterContainer.addEventListener("touchstart", (e) => {

            beforeAfterContainer.addEventListener("touchmove", (e2) => {
                let containerWidth = beforeAfterContainer.offsetWidth;
                let currentPoint = e2.changedTouches[0].clientX;

                let startOfDiv = beforeAfterContainer.offsetLeft;

                let modifiedCurrentPoint = currentPoint - startOfDiv;

                if (modifiedCurrentPoint > 10 && modifiedCurrentPoint < beforeAfterContainer.offsetWidth - 10) {
                    let newWidth = modifiedCurrentPoint * 100 / containerWidth;

                    before.setAttribute('style', "width:" + newWidth + "%;");
                    afterText.setAttribute('style', "z-index: 1;");
                    handle.setAttribute('style', "left:" + newWidth + "%;");
                }
            });
        });

        //mouse move event listener
        beforeAfterContainer.addEventListener('mousemove', (e) => {
            let containerWidth = beforeAfterContainer.offsetWidth;
            widthChange = e.offsetX;
            let newWidth = widthChange * 100 / containerWidth;

            if (e.offsetX > 10 && e.offsetX < beforeAfterContainer.offsetWidth - 10) {
                before.setAttribute('style', "width:" + newWidth + "%;");
                afterText.setAttribute('style', "z-index:" + "1;");
                handle.setAttribute('style', "left:" + newWidth + "%;");
            }
        })

    }
}

class MultiCompare {
    constructor(entry) {
        const container = document.querySelector(entry.id);
        if (!container) return;

        const handles = Array.from(container.querySelectorAll('.mc-handle'));
        const layers = [
            container.querySelector('.mc-layer-1'),
            container.querySelector('.mc-layer-2'),
            container.querySelector('.mc-layer-3')
        ];

        const minGap = 5;
        let positions = (entry.positions && entry.positions.length === 3)
            ? entry.positions.slice()
            : [25, 50, 75];

        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

        const applyPositions = () => {
            layers[0].style.width = positions[0] + '%';
            layers[1].style.width = positions[1] + '%';
            layers[2].style.width = positions[2] + '%';
            handles.forEach((handle, idx) => {
                handle.style.left = positions[idx] + '%';
            });
        };

        const updateFromPointer = (index, clientX) => {
            const rect = container.getBoundingClientRect();
            const raw = ((clientX - rect.left) / rect.width) * 100;
            const leftBound = index === 0 ? minGap : positions[index - 1] + minGap;
            const rightBound = index === positions.length - 1
                ? 100 - minGap
                : positions[index + 1] - minGap;
            positions[index] = clamp(raw, leftBound, rightBound);
            applyPositions();
        };

        const startDrag = (index, event, captureTarget) => {
            captureTarget.setPointerCapture(event.pointerId);

            const onMove = (moveEvent) => updateFromPointer(index, moveEvent.clientX);
            const onUp = () => {
                captureTarget.releasePointerCapture(event.pointerId);
                captureTarget.removeEventListener('pointermove', onMove);
                captureTarget.removeEventListener('pointerup', onUp);
            };

            captureTarget.addEventListener('pointermove', onMove);
            captureTarget.addEventListener('pointerup', onUp);
        };

        handles.forEach((handle) => {
            handle.addEventListener('pointerdown', (event) => {
                event.stopPropagation();
                const index = Number(handle.dataset.index);
                startDrag(index, event, handle);
            });
        });

        container.addEventListener('pointerdown', (event) => {
            const rect = container.getBoundingClientRect();
            const percent = ((event.clientX - rect.left) / rect.width) * 100;
            let nearestIndex = 0;
            let nearestDistance = Math.abs(percent - positions[0]);

            for (let i = 1; i < positions.length; i += 1) {
                const distance = Math.abs(percent - positions[i]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestIndex = i;
                }
            }

            startDrag(nearestIndex, event, container);
        });

        window.addEventListener('resize', applyPositions);
        applyPositions();
    }
}
