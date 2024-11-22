import MenuElement from "./MenuElement/MenuElement";
import "./MenuContainer.css";

export default function MenuContainer() {
    return (
        <>
            <div className="MenuBackground">
                <MenuElement className="Title"><h2>Performance Metrics</h2></MenuElement>
                <div className="MenuContainer">
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                </div>
                <div className="MenuContainer">
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                </div>
                <div className="MenuContainer">
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                    <MenuElement className="MenuElement"></MenuElement>
                </div>
            </div>

        </>
    );
}