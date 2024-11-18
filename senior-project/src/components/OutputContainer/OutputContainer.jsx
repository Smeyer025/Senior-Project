/**
 * OutputContainer Component: Container for output
 */

import "./OutputContainer.css";

export default function OutputContainer(props) {
    const determineColor = (children) => {
        console.log(`children: ${children}`);
        if (children == "positive") {
            return "green";
        } else if (children == "negative") {
            return 'red';
        } else {
            return 'black';
        }
    };

    return (
        <div className="OC">
            <h1 style={{color: determineColor(props.children)}}>{props.children}</h1>
        </div>
    );
};